import numpy as np
import cv2 as cv
from math import sqrt, acos, degrees, sin, cos, radians


#####! THIS VERSION OF THE PROGRAM FROM THE SECONDS ASSIGNMENT INCLUDES SOME UPDATES:
#####!      ^ Avoids creating the video, it just produces the csv file.
#####!      ^ Avoids the header (column names) of the csv table.


def retrieve_rotated_labels() -> None:
    """Return a dictionary which maps every possible id/label
    of the turn table to the related 3D coordinates.
    """

    def rotate_position(id: int) -> tuple[float, float]:
        """Compute the rotation of the given label.

        Args:
            id (int): marker label/id

        Returns:
            tuple[float, float]: matching coordinates.
        """
        angle15 = radians(-15)
        qx = round(cos(angle15 * id) * 70, 1)
        qy = round(sin(angle15 * id) * 70, 1)

        return qx, qy

    return {marker_id: rotate_position(marker_id) for marker_id in range(0, 24)}


def dist2D(one: list[float], two: list[float]) -> float:
    """Computes the 2D euclidean distance between the given
    points.

    Args:
        one (list[float]): point number 1.
        two (list[float]): point number 2.

    Returns:
        float: computed distance.
    """
    dx = one[0] - two[0]
    dy = one[1] - two[1]
    return sqrt(dx * dx + dy * dy)


def angle3P(p1: list[int], p2: list[int], p3: list[int]) -> float:
    """Computes the angle between 3 points, where p3 will be the
    central point (the one which "hosts" the angle).

    Args:
        p1 (list[int]): point 1.
        p2 (list[int]): point 2.
        p3 (list[int]): point 3 (middle point).

    Returns:
        float: computed angle (in degrees).
    """
    a = dist2D(p3, p1)
    b = dist2D(p3, p2)
    c = dist2D(p1, p2)

    numer = c**2 - a**2 - b**2
    denom = -2 * a * b
    if denom == 0:
        denom = 0.000001
    rads = acos(numer / denom)
    degs = degrees(rads)

    if p1[1] > p3[1]:
        degs = 360 - degs
    return degs


def find_concave_angle(cv_points: np.ndarray) -> int:
    """Find the index of the concave angle in the given
    array of points.

    Args:
        cv_points (np.ndarray): array of points.

    Returns:
        int: index of concave angle.
    """
    points = cv_points.reshape(-1, 2)

    a = points - np.roll(points, 1, axis=0)
    b = np.roll(a, -1, axis=0)

    return np.argmin(np.cross(a, b))


def sort_contours_clockwise(contours: np.ndarray) -> np.ndarray:
    """Sort the given contours clockwise, by fitting
    the smallest possible circle around all their
    centres, and sorting them through their angles.
    Centres are used because they allow to establish
    with enough precision (for this use case) the
    direction of each contour/polygon.


    Args:
        contours (np.ndarray): array of contours.

    Returns:
        np.ndarray: sorted array of contours.
    """
    centroids = []
    centers = []

    for cnt in contours:
        m = cv.moments(cnt)
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        centers.append([cx, cy])
        centroids.append([[cx, cy], cnt])

    numped = np.array(centers)
    (x, y), _ = cv.minEnclosingCircle(numped)
    middle = [x, y]
    offshoot = [x + 100, y]

    angles = [[angle3P(cen[0], offshoot, middle), cen[0], cen[1]] for cen in centroids]

    final = sorted(angles, key=lambda a: a[0], reverse=True)

    contours = [clump[2] for clump in final]

    return contours


def check_continuity(markers_dictionary: list, current_index: int) -> bool:
    """Check whether the next marker is the real successor of the
    current marker.
    A real successor should be more distant than 120-y circa from
    the current marker.

    Args:
        markers_dictionary (list): list of markers.
        current_index (int): index of current marker.

    Returns:
        bool: True if the next marker is the real successor,
        false otherwise (i.e., the real successor is occluded).
    """
    act_y = markers_dictionary[current_index][0][0][0][0][1]
    next_y = markers_dictionary[(current_index + 1) % len(markers_dictionary)][0][0][0][
        0
    ][1]
    diff_y = abs(next_y - act_y)
    return diff_y < 200


def indetify_single_circle_marker(markers_dictionary: list, current_index: int) -> int:
    """Identify the current single-circle marker. A single circle marker followed
    by 5 cirlces is a 23, the one followed by 4 is a 15.

    Args:
        markers_dictionary (list): list of marker-circles.
        current_index (int): current single-circle marker index.

    Returns:
        int: current marker id.
    """
    return (
        23
        if len(markers_dictionary[(current_index + 1) % len(markers_dictionary)][0]) - 1
        == 5
        else 15
    )


def single_circle_based_marker_detection(
    gray_copy: np.ndarray,
) -> dict[tuple[int, int], int]:
    """Detect all markers' and circles' polygons, sort the
    markers' contours/polygons clockwise, and look for the
    marker wich has only one circle and has a visible successor
    (i.e., not covered by the glass).
    Identify the single-circle marker (15 or 23), and start
    indetifying all the markers from it.

    Args:
        gray_copy (np.ndarray): grayed frame (already cropped).

    Returns:
        dict[tuple[int,int], int]: dictionary mapping the concave corner
        of each detected/identified marker to the related label.
    """
    _, threshold = cv.threshold(gray_copy, 194, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    marker_contours = sort_contours_clockwise(
        [
            cnt
            for cnt in contours
            if len(cv.approxPolyDP(cnt, 0.016 * cv.arcLength(cnt, True), True)) == 5
            and cv.contourArea(cnt) > 1500
        ],
    )

    circle_polys = [
        act_poly
        for cnt in contours
        if len(act_poly := cv.approxPolyDP(cnt, 0.016 * cv.arcLength(cnt, True), True))
        > 5
    ]

    marker_polys = [
        cv.approxPolyDP(marker_cnt, 0.016 * cv.arcLength(marker_cnt, True), True)
        for marker_cnt in marker_contours
    ]

    markers_dictionary = [
        [
            [act_marker]
            + [
                act_circle
                for act_circle in circle_polys
                if cv.pointPolygonTest(
                    act_marker,
                    (
                        int(act_circle[0][0][0]),
                        int(act_circle[0][0][1]),
                    ),
                    False,
                )
                >= 0
            ]
        ]
        for act_marker in marker_polys
    ]

    point_to_id_map = {}
    for mac_index, marker_and_circles in enumerate(markers_dictionary):
        if len(marker_and_circles[0]) - 1 == 1 and check_continuity(
            markers_dictionary, mac_index
        ):
            labeling_visit_index = mac_index
            num_of_visible_markers = len(markers_dictionary)

            last_label = indetify_single_circle_marker(
                markers_dictionary=markers_dictionary,
                current_index=mac_index,
            )
            is_clockwise = True
            single_circle_index = mac_index

            single_circle_label = last_label
            last_label -= 1
            for _ in range(0, num_of_visible_markers):
                last_label = (last_label + (1 if is_clockwise else -1)) % 24

                current_marker_corners = markers_dictionary[labeling_visit_index][0][0]

                index_of_concave = find_concave_angle(current_marker_corners)

                tupled_point = (
                    current_marker_corners[index_of_concave][0][0],
                    current_marker_corners[index_of_concave][0][1],
                )

                point_to_id_map[tupled_point] = last_label

                right_continuity = (
                    check_continuity(markers_dictionary, labeling_visit_index)
                    if is_clockwise
                    else True
                )
                if not right_continuity:
                    labeling_visit_index = (
                        single_circle_index - 1
                    ) % num_of_visible_markers
                    is_clockwise = False
                    last_label = single_circle_label
                else:
                    labeling_visit_index = (
                        labeling_visit_index + (1 if is_clockwise else -1)
                    ) % num_of_visible_markers
            return point_to_id_map
    return point_to_id_map


def marker_identification_and_tracking(video_index: int) -> None:
    """Identify the markers through the function single_circle_based_marker_detection,
    and then keep tracking their positions, refreshing them by calling again the
    detection function after a number of frames.

    Args:
        video_index (int): index of the chosen video.
    """
    video_path = f"data/obj0{video_index}.mp4"
    with open(f"data/obj0{video_index}_marker.csv", "w") as fd:
        fd.write("")

    vidcap = cv.VideoCapture(video_path)
    success, frame = vidcap.read()

    num_of_frames = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))

    lk_params = dict(
        winSize=(20, 20),
        maxLevel=1,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    out_w = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
    CROP_START, CROP_END = int(out_w / 1.641), int(out_w / 1.22)
    old_gray_frame = cv.cvtColor(frame[:, CROP_START:CROP_END], cv.COLOR_BGR2GRAY)

    csv_row = ""

    rotated_labels = retrieve_rotated_labels()

    for frame_index in range(num_of_frames):
        if success:
            current_gray_frame = cv.cvtColor(
                frame[:, CROP_START:CROP_END], cv.COLOR_BGR2GRAY
            )
            if frame_index % 12 == 0:
                previous_frame_dict_temp = single_circle_based_marker_detection(
                    gray_copy=current_gray_frame
                )
                previous_frame_dict = (
                    previous_frame_dict_temp
                    if previous_frame_dict_temp != {}
                    else previous_frame_dict
                )
                p0 = np.array(
                    [
                        np.array(
                            [(lab_point_tuple[0], lab_point_tuple[1])], dtype=np.float32
                        )
                        for lab_point_tuple in previous_frame_dict
                    ],
                    dtype=np.float32,
                )

            p1, st, _ = cv.calcOpticalFlowPyrLK(
                old_gray_frame, current_gray_frame, p0, None, **lk_params
            )

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            current_frame_dict = {}
            for _, (current_point, old_point) in enumerate(zip(good_new, good_old)):
                a, b = current_point.astype(int)
                c, d = old_point.astype(int)
                label = previous_frame_dict[(c, d)]
                current_frame_dict[(a, b)] = label
                corrected_a = a + CROP_START
                csv_row += (
                    f"{frame_index},{label},{corrected_a},"
                    f"{b},{rotated_labels[label][0]},{rotated_labels[label][1]},0\n"
                )

                frame = cv.putText(
                    img=frame,
                    text=str(label),
                    org=(corrected_a, b),
                    fontScale=1.3,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    color=(0, 0, 0),
                    thickness=10,
                )

                frame = cv.putText(
                    img=frame,
                    text=str(label),
                    org=(corrected_a, b),
                    fontScale=1.3,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    color=(255, 255, 255),
                    thickness=3,
                )

                frame = cv.circle(
                    frame,
                    (corrected_a, b),
                    radius=5,
                    color=(0, 255, 0),
                    thickness=3,
                )

            previous_frame_dict = current_frame_dict

            old_gray_frame = current_gray_frame
            p0 = good_new.reshape(-1, 1, 2)

            success, frame = vidcap.read()
        else:
            break
    vidcap.release()

    with open(f"data/obj0{video_index}_marker.csv", "a") as fd:
        fd.write(csv_row)
