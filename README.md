## Installation

This package depends on a recent version of OpenCV python library and transforms libraries:

```bash
$ pip3 install opencv-python opencv-contrib-python transforms3d

$ sudo apt install ros-iron-tf-transformations
```

Build the package from source with `colcon build --symlink-install` in the workspace root.


## Running Marker Detection for Pose Estimation

Launch the aruco pose estimation node with this command. The parameters will be loaded from _aruco\_parameters.yaml_,
but can also be changed directly in the launch file with command line arguments.

```bash
ros2 launch aruco_pose_estimation aruco_pose_estimation.launch.py
```

