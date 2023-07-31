import numpy as np
import cv2 as cv
from detection_and_tracking import marker_identification_and_tracking
from os.path import isfile
from concurrent.futures import ThreadPoolExecutor as Pool


def read_coordinates_from_csv(filename: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Reads the content of the csv file found in the given path and groups the 2D and
    3D coordinates by frame index.

    Args:
        filename (str): relative path of the csv file.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: List of tuples of two numpy arrays: the
        first one contains the 2D coordinates, the second contains the corresponding
        3D coordinates.
    """
    # Load the CSV file into a NumPy array.
    data = np.loadtxt(filename, delimiter=",", usecols=(0, 2, 3, 4, 5, 6))

    # Convert the first column to integers.
    data[:, 0] = data[:, 0].astype(int)

    # Split the data into groups by frame index.
    groups_by_frame = {}
    for row in data:
        frame_index = row[0]
        if frame_index not in groups_by_frame:
            groups_by_frame[frame_index] = [[], []]
        groups_by_frame[frame_index][0].append([row[1], row[2]])
        groups_by_frame[frame_index][1].append([row[3], row[4], row[5]])

    return [
        (
            np.array(coords[0], dtype=np.float32),  # 2D
            np.array(coords[1], dtype=np.float32),  # 3D
        )
        for coords in groups_by_frame.values()
    ]


# --- GLOBAL VARIABLES SHARED BY THREADS ---

# Camera intrinsic parameters.
K = np.array(
    [
        [1.66750771e03, 0.00000000e00, 9.54599045e02],
        [0.00000000e00, 1.66972683e03, 5.27926123e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ],
    dtype=np.float32,
)

# Distortion coefficient.
dist_coef = np.array(
    [
        1.16577217e-01,
        -9.28944623e-02,
        7.15149511e-05,
        -1.80025974e-03,
        -1.24761932e-01,
    ],
    dtype=np.float32,
)

# Indices of the vertices that form each face of the cube.
cube_faces = np.array(
    [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
    ],
    dtype=np.int8,
)

# Define the 3D coordinates of the cube in object space.
cube_pts = np.array(
    [
        (70.0, 0.0, 75.0),
        (0.0, -70.0, 75.0),
        (-70.0, 0.0, 75.0),
        (0.0, 70.0, 75.0),
        (70.0, 0.0, 173.99495),
        (0.0, -70.0, 173.99495),
        (-70.0, 0.0, 173.99495),
        (0.0, 70.0, 173.99495),
    ],
    dtype=np.float32,
)

all_frames_and_coords = None

# --- --- --- --- --- --- --- --- --- --- ---


def project_cube_over_frame(index: int) -> np.ndarray:
    """Use the given 2D and 3D coordinates to project a cube over the undistorted
    frame according to the obtained pose.

    Args:
        index (int): index of the structure that contains frame and related
        coordinates.

    Returns:
        np.ndarray: modified frame that includes the cube.
    """
    # Access the required frame and coordinates.
    global all_frames_and_coords
    frame, points_2d, points_3d = (
        # Transform the frame to compensate for lens distortion using k and dist_coef
        cv.undistort(all_frames_and_coords[index][0], K, dist_coef),
        all_frames_and_coords[index][1][0],
        all_frames_and_coords[index][1][1],
    )
    # Set the cell to None to hint the GC.
    all_frames_and_coords[index] = None

    # Finds the object pose (rotation and the translation vectors) from 3D-2D
    # point correspondences.
    _, rvec, tvec = cv.solvePnP(
        points_3d, points_2d, K, dist_coef, flags=cv.SOLVEPNP_IPPE
    )

    # Project the 3D points in 2D using the computed pose.
    cube_pts_2d, _ = cv.projectPoints(cube_pts, rvec, tvec, K, dist_coef)

    # Draw the projected points onto the frame.
    cv.polylines(
        frame,
        [cube_pts_2d[face].reshape((-1, 2)).astype(int) for face in cube_faces],
        True,
        (0, 255, 0),
        thickness=3,
    )

    return frame


def project_cube_over_frames(video_index: int, basic_path: str) -> None:
    """Read frames and coordinates and assign them to a pool of threads which will
    process each frame with the "project_cube_over_frame" function, storing the produced
    frames into a video of the same format of the input one.

    Args:
        video_index (int): index of the video to process.
        basic_path (str): basic path pointing to the location of the input/output data folder
        that will include the input videos and the csv files.
    """
    global all_frames_and_coords
    # Init variables needed to read the frames from the chosen video.
    video_path = f"{basic_path}{video_index}.mp4"
    vidcap = cv.VideoCapture(video_path)
    num_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

    (out_h, out_w) = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(
        vidcap.get(cv.CAP_PROP_FRAME_WIDTH)
    )
    fps = vidcap.get(cv.CAP_PROP_FPS)

    # Read coordinates and frames.
    coords_by_frame_dict = read_coordinates_from_csv(
        filename=f"{basic_path}{video_index}_marker.csv"
    )

    # Read the frames and couple each of them with the related
    # coordinates to produce and easily accessible structure for
    # multithreaded processing.
    all_frames_and_coords = np.empty((num_frames), dtype=object)
    for i in range(num_frames):
        ret, frame = vidcap.read()
        if not ret:
            break
        all_frames_and_coords[i] = (
            frame,
            coords_by_frame_dict[i],
        )
    vidcap.release()
    del coords_by_frame_dict

    with Pool() as pool:
        # Parallelize the projection of cubes over each frame.
        results = pool.map(
            project_cube_over_frame,
            range(num_frames),
            chunksize=2,
        )
        # Speed improvement: ~2.5 on octa-core hw.
        # Bottleneck: sequential image writing:
        #    30% of time is used for data read and image processing.
        #    70% of time is used for images writing.

        out = cv.VideoWriter(
            f"{basic_path}{video_index}_cube.mp4",
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )

        for processed_frame in results:
            out.write(processed_frame)

        out.release()


if __name__ == "__main__":
    chosen_video = int(
        input(
            "Choose which video to process [1 Toucan, 2 Dinosaur, 3 Cracker, 4 Statue]:"
        )
    )
    while chosen_video not in [1, 2, 3, 4]:
        chosen_video = int(
            input(
                "Choose which video to process [1 Toucan, 2 Dinosaur, 3 Cracker, 4 Statue]:"
            )
        )

    # chosen_video = 1  --DEBUG

    basic_path = "./data/obj0"

    if not isfile(f"{basic_path}{chosen_video}_marker.csv"):
        print("----CSV file missing! Starting creation----\n")
        marker_identification_and_tracking(chosen_video)
        print("----CSV file created.----\n")
    else:
        print("----CSV file already existing!----\n")

    print("----Starting cube projection----")
    project_cube_over_frames(chosen_video, basic_path)
    print("----Completed with success----")
