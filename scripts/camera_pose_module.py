# camera_pose_module.py

import numpy as np
import cv2

def estimate_camera_pose_from_markers(
    marker_corners_2d_list,
    marker_corners_3d_list,
    camera_matrix,
    dist_coeffs=None,
    flags=cv2.SOLVEPNP_ITERATIVE
):
    """
    Perform a single solvePnP using all visible marker corners:
    
    Args:
        marker_corners_2d_list: list of 2D points from all markers (shape Nx2).
        marker_corners_3d_list: list of 3D object points in the marker frame (shape Nx3).
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: distortion coefficients, if any. Default None.
        flags: solvePnP flags (e.g. cv2.SOLVEPNP_ITERATIVE or cv2.SOLVEPNP_AP3P etc.)

    Returns:
        rvec, tvec (each shape (3,1)) describing the camera pose
        in the "marker" or "board" coordinate system.
    """
    if len(marker_corners_2d_list) < 4:
        # Not enough points to solvePnP robustly
        return None, None

    # Convert to the correct shape for solvePnP
    obj_points = np.array(marker_corners_3d_list, dtype=np.float32)  # Nx3
    img_points = np.array(marker_corners_2d_list, dtype=np.float32)  # Nx2

    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points,
        camera_matrix, dist_coeffs,
        flags=flags
    )
    if not success:
        return None, None

    return rvec, tvec


def compute_camera_pose(rvec, tvec):
    """
    Convert the marker-based pose (rvec, tvec) into a camera pose.

    Parameters:
        rvec (numpy.ndarray): Rotation vector from solvePnP (shape 3x1).
        tvec (numpy.ndarray): Translation vector from solvePnP (shape 3x1).

    Returns:
        tuple:
            - R_camera (numpy.ndarray): 3x3 rotation matrix representing the camera's orientation in the world frame.
            - T_camera (numpy.ndarray): 3x1 translation vector representing the camera's position in the world frame.

    Transformation:
        Given a point in the camera frame (p_cam), its corresponding point in the world (marker) frame (p_world) is:
            p_world = R_camera * p_cam + T_camera

        Where:
            R_camera = R_marker.T
            T_camera = -R_marker.T @ tvec

        This ensures that p_world accurately represents the position in the global coordinate system.
    """
    R_marker, _ = cv2.Rodrigues(rvec)  # rotation from marker -> camera
    R_camera = R_marker.T             # camera -> marker
    T_camera = -R_marker.T @ tvec
    return R_camera, T_camera

