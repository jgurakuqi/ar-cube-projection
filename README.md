# ar-cube-projection

## Table of Contents

- [Description](#Description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Description
The goal of the project/assignment is to extract the pose of an object in a given video, given the 2D and 3D coordinates of reference markers per each 
frame, and project a virtual cube over each frame using the estimated poses. The project includes also an extra python script which allows to extract
the 3D and 2D coordinates for a specific problem, hence useful only for the given dataset.
The said coordinates are provided through a CSV file.
The project implements multi-threading for the main task, allowing to achieve optimal performance, especially with long videos.


## Installation

In order to run this project it's required a Python 3 installation, along with OpenCV and numpy. For the 2 libraries installation the following two *pip* 
commands can be used:
```bash
pip install opencv-python
pip install numpy
```

## Usage

To run the projects it's required an IDE or a bash/shell. In case of a bash/shell, it's sufficient to use the following command:

```bash
python augmented_reality.py
```

If the file containing the 2D-3D coordinates used by the main program doesn't exist, the main will automatically invoke the *marker_identification_and_tracking* function
from the *detection_and_tracking.py* script to produce such CSV file.
The 2 python scripts hardcode the input videos' format as mp4.

## Contributing

This project allows to extract from a video the pose of an object given a set of object points, their corresponding image projections, as well as the camera intrinsic 
matrix and the distortion coefficients, and to project the chosen 3D-world points representing a cube over the said input video.
This project could be generalized to allow the use of different shapes and video formats.

```bash
git clone https://github.com/jgurakuqi/ar_cube_projection
```

As for the 3D-2D coordinates, that is a problem-specific task, which requires some analysis, determining which points to extract and how to do it in the most appropriate way.

## License

MIT License

Copyright (c) 2023 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
