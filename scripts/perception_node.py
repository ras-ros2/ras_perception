#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# For the Trigger service (if you need more arguments, define a custom service)
from std_srvs.srv import Trigger

import pyrealsense2 as rs
import cv2
import numpy as np
import math
import os
import yaml
import time

# -------------------------------------------------------------------------
# Your previously imported modules, exactly as before
# -------------------------------------------------------------------------
from plane_module import fit_plane_to_points_ransac
from camera_pose_module import (
    estimate_camera_pose_from_markers,
    compute_camera_pose
)
from intersection_module import intersect_ray_with_plane
from yolo_module import (
    load_yolo_model,
    detect_objects,
    median_depth_in_bbox_center,
    compute_tictactoe_cell_centers_2d
)

# -------------------------------------------------------------------------
# Global settings / constants (unchanged)
# -------------------------------------------------------------------------
YAML_FILE = r"Downloads\markers.yaml"
DISTANCE_THRESHOLD = 0.04
MARKER_LENGTH      = 0.04
OCCLUSION_RATIO    = 0.8

# RealSense camera intrinsics
FX = 921.1829
FY = 931.1829
CX = 640
CY = 360

camera_matrix = np.array([
    [FX,    0,   CX],
    [ 0,   FY,   CY],
    [ 0,    0,    1]
], dtype=np.float32)

dist_coeffs = None

# YOLO model path
YOLO_MODEL_PATH = r"C:\Users\saman\Downloads\tictactoe_model_updated.pt"

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_params = cv2.aruco.DetectorParameters()

# -------------------------------------------------------------------------
# YAML helpers for inventory (unchanged)
# -------------------------------------------------------------------------
def _load_yaml():
    if not os.path.exists(YAML_FILE):
        return {"markers": {}}
    with open(YAML_FILE, 'r') as f:
        data = yaml.safe_load(f) or {}
    if "markers" not in data:
        data["markers"] = {}
    return data

def _save_yaml(data):
    with open(YAML_FILE, 'w') as f:
        yaml.dump(data, f)

# -------------------------------------------------------------------------
# Marker labeling / size
# -------------------------------------------------------------------------
def get_aruco_3d_corners(marker_length):
    half = marker_length / 2.0
    return np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

# -------------------------------------------------------------------------
# Inventory-like functions (unchanged)
# -------------------------------------------------------------------------
def initialize(marker_positions):
    """
    Detect black/white markers, label them w1.., b1.., store to markers.yaml
    """
    data = {"markers": {}}

    white_list = marker_positions["white"]
    black_list = marker_positions["black"]

    w_count = 1
    b_count = 1
    for pos in white_list:
        marker_id = f"w{w_count}"
        data["markers"][marker_id] = {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        }
        w_count += 1

    for pos in black_list:
        marker_id = f"b{b_count}"
        data["markers"][marker_id] = {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2])
        }
        b_count += 1

    _save_yaml(data)
    print("[INFO] Initialize: wrote w/b markers to", YAML_FILE)

def update_markers(marker_positions):
    """
    Re-detect marker positions.
    """
    data = _load_yaml()
    old_markers = data["markers"]

    new_white = marker_positions["white"]
    new_black = marker_positions["black"]

    new_detections = {
        "white": new_white.copy(),
        "black": new_black.copy()
    }
    matched_detections = {
        "white": [],
        "black": []
    }

    moved_markers = []

    # Iterate over old markers
    for m_id, coords in old_markers.items():
        marker_type = 'white' if m_id.startswith('w') else 'black'
        old_pos = np.array([coords["x"], coords["y"], coords["z"]])

        # Initialize as moved
        moved = True
        new_pos = None

        # Check for any detection within the safe zone
        for idx, det_pos in enumerate(new_detections[marker_type]):
            det_pos_np = np.array(det_pos)
            distance = np.linalg.norm(det_pos_np - old_pos)
            if distance <= DISTANCE_THRESHOLD:
                # Match found
                moved = False
                new_pos = det_pos
                matched_detections[marker_type].append(idx)
                break

        if not moved and new_pos is not None:
            data["markers"][m_id] = {
                "x": float(new_pos[0]),
                "y": float(new_pos[1]),
                "z": float(new_pos[2])
            }
        elif moved:
            # Marker is considered moved. Look for a new detection outside all safe zones
            for idx, det_pos in enumerate(new_detections[marker_type]):
                if idx in matched_detections[marker_type]:
                    continue
                det_pos_np = np.array(det_pos)
                outside_all_safe_zones = True
                for other_m_id, other_coords in old_markers.items():
                    if other_m_id == m_id:
                        continue
                    other_pos = np.array([other_coords["x"], other_coords["y"], other_coords["z"]])
                    if np.linalg.norm(det_pos_np - other_pos) <= DISTANCE_THRESHOLD:
                        outside_all_safe_zones = False
                        break

                if outside_all_safe_zones:
                    new_pos = det_pos
                    data["markers"][m_id] = {
                        "x": float(new_pos[0]),
                        "y": float(new_pos[1]),
                        "z": float(new_pos[2])
                    }
                    matched_detections[marker_type].append(idx)
                    moved_markers.append((m_id, new_pos))
                    break

    _save_yaml(data)

    if moved_markers:
        moved_info = "\n".join([f"Marker '{m_id}' moved to {tuple(pos)}" for m_id, pos in moved_markers])
        print(f"[INFO] Updated markers:\n{moved_info}")
    else:
        print("[INFO] No significant marker movement found.")

    return moved_markers

def fetch(marker_id):
    """
    Return (x, y, z) for a marker from markers.yaml
    """
    data = _load_yaml()
    markers = data.get("markers", {})
    if marker_id not in markers:
        return None
    m = markers[marker_id]
    return (m["x"], m["y"], m["z"])

# -------------------------------------------------------------------------
# Tic-Tac-Toe grid occupancy (unchanged, except no GUI calls)
# -------------------------------------------------------------------------
def compute_grid_occupancy():
    """
    Compute which cells of the Tic Tac Toe grid are occupied
    based on current marker positions in markers.yaml.

    Returns:
        dict: Keys are cell labels (e.g. 'A1'), values are marker IDs (e.g. 'w1', 'b2') or '-' if empty.
    """
    # Define frame center (assuming your intrinsics for rough alignment)
    frame_center_px = (CX, CY)

    # Compute cell centers in 2D (from your yolo_module)
    cell_centers_2d = compute_tictactoe_cell_centers_2d(
        frame_center_px,
        dist_edge_px=80,
        dist_corner_px=105,
        margin_scale=1.8
    )

    # Define cell labels
    cell_labels = ['A1', 'A2', 'A3',
                   'B1', 'B2', 'B3',
                   'C1', 'C2', 'C3']

    # Initialize occupancy dictionary
    grid_occupancy = {label: '-' for label in cell_labels}

    # Load current markers from YAML
    data = _load_yaml()
    markers = data.get("markers", {})

    # Project marker positions into 2D
    marker_pixel_positions = {}
    for m_id, coords in markers.items():
        x, y, z = coords["x"], coords["y"], coords["z"]
        if z == 0:
            continue
        u = (x * FX) / z + CX
        v = (y * FY) / z + CY
        marker_pixel_positions[m_id] = (u, v)

    # Threshold for marking occupancy
    pixel_threshold = 50

    # Assign markers to grid cells
    cc2d_flat = cell_centers_2d.flatten().reshape(-1, 2)
    for label, center in zip(cell_labels, cc2d_flat):
        cell_u, cell_v = center
        for m_id, (marker_u, marker_v) in marker_pixel_positions.items():
            distance = math.hypot(marker_u - cell_u, marker_v - cell_v)
            if distance <= pixel_threshold:
                grid_occupancy[label] = m_id
                break  # only assign first marker found

    return grid_occupancy

# -------------------------------------------------------------------------
# Single-pass detection
# -------------------------------------------------------------------------
def run_detection_once(color_image, depth_image, depth_scale):
    corners_2d, ids, _ = cv2.aruco.detectMarkers(
        color_image, aruco_dict, parameters=aruco_params
    )
    rvec, tvec = None, None
    marker_positions = {"black": [], "white": []}
    points_with_tags = []

    # SolvePnP if at least one marker is found
    if ids is not None and len(ids) > 0:
        min_id_index = np.argmin(ids)
        corners_min = corners_2d[min_id_index][0]
        marker_3d = get_aruco_3d_corners(MARKER_LENGTH)

        single_objpoints = []
        single_imgpoints = []
        for j in range(4):
            single_objpoints.append(marker_3d[j])
            single_imgpoints.append(corners_min[j])

        single_objpoints = np.array(single_objpoints, dtype=np.float32)
        single_imgpoints = np.array(single_imgpoints, dtype=np.float32)

        rvec, tvec = estimate_camera_pose_from_markers(
            single_imgpoints,
            single_objpoints,
            camera_matrix,
            dist_coeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    # Default plane + camera
    plane_normal = np.array([0, 0, 1], dtype=float)
    plane_point = np.zeros(3, dtype=float)
    R_camera = np.eye(3, dtype=float)
    T_camera = np.zeros(3, dtype=float)

    # If we found a valid pose
    if rvec is not None and tvec is not None:
        R_camera, T_camera = compute_camera_pose(rvec, tvec)
        T_camera = T_camera.flatten()

        # Collect corners for plane fitting
        for i_m in range(len(ids)):
            tag_id = ids[i_m][0]
            c2d = corners_2d[i_m][0]
            for j in range(4):
                u = int(c2d[j, 0])
                v = int(c2d[j, 1])
                raw_d = median_depth_in_bbox_center(depth_image, u, v, half_window=1)
                d_meters = raw_d * depth_scale
                if d_meters <= 0:
                    continue
                X = (u - CX) / FX * d_meters
                Y = (v - CY) / FY * d_meters
                Z = d_meters
                point = np.array([X, Y, Z], dtype=float)
                points_with_tags.append((tag_id, point))

        # Fit plane if enough points
        if len(points_with_tags) >= 3:
            plane_normal, plane_point = fit_plane_to_points_ransac(points_with_tags)
        else:
            plane_normal = np.array([0, 0, 1], dtype=float)
            plane_point = np.array([0, 0, 0], dtype=float)

    # YOLO detection for black/white markers
    boxes = detect_objects(model, color_image, conf=0.5)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        # cls=0 => black, cls=2 => white
        if cls not in [0, 2]:
            continue

        u_center = int((x1 + x2) / 2)
        v_center = int((y1 + y2) / 2)

        intersection_cam = intersect_ray_with_plane(
            u_center, v_center,
            plane_normal, plane_point,
            FX, FY, CX, CY
        )
        if intersection_cam is None:
            continue

        intersection_cam = intersection_cam.flatten()
        intersection_global = R_camera.T @ intersection_cam + T_camera

        if cls == 0:
            marker_positions["black"].append(tuple(intersection_global))
        else:
            marker_positions["white"].append(tuple(intersection_global))

    return (
        marker_positions,
        plane_normal,
        plane_point,
        R_camera,
        T_camera,
        corners_2d,
        ids,
        rvec,
        tvec
    )

# -------------------------------------------------------------------------
# The ROS 2 Perception Node (NO GUI)
# -------------------------------------------------------------------------
class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.get_logger().info(f"[INFO] Depth Scale is: {self.depth_scale}")

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # We'll store the latest marker positions from detection
        self.marker_positions = {"black": [], "white": []}

        # Timer: capture frames & run detection at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Create services
        self.srv_init = self.create_service(
            Trigger,
            'initialize_markers',
            self.initialize_markers_callback
        )
        self.srv_update = self.create_service(
            Trigger,
            'update_markers',
            self.update_markers_callback
        )

        # OPTIONAL: If you want to fetch a marker's position by ID via service, uncomment:
        # self.srv_fetch = self.create_service(
        #     MyCustomFetchSrv,  # define a custom service with a string `marker_id` and a float32[] response
        #     'fetch_marker',
        #     self.fetch_marker_callback
        # )
        #
        # OPTIONAL: If you want to compute & return TicTacToe occupancy by service, do similarly.

        self.get_logger().info("PerceptionNode started, services ready.")

    def timer_callback(self):
        """
        Periodic callback: get frames, run detection, store marker_positions.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        (marker_positions,
         plane_normal,
         plane_point,
         R_camera,
         T_camera,
         corners_2d,
         ids,
         rvec,
         tvec) = run_detection_once(
            color_image, depth_image, self.depth_scale
        )

        self.marker_positions = marker_positions

    # -----------------------------
    # Service callbacks
    # -----------------------------
    def initialize_markers_callback(self, request, response):
        """
        /initialize_markers service calls your original initialize() function.
        """
        if not self.marker_positions["black"] and not self.marker_positions["white"]:
            msg = "[ERROR] No markers detected. Cannot initialize."
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        initialize(self.marker_positions)
        msg = "[INFO] Markers have been initialized and saved to markers.yaml."
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def update_markers_callback(self, request, response):
        """
        /update_markers service calls your original update_markers() function.
        """
        if not self.marker_positions["black"] and not self.marker_positions["white"]:
            msg = "[ERROR] No markers detected. Cannot update."
            self.get_logger().error(msg)
            response.success = False
            response.message = msg
            return response

        moved_markers = update_markers(self.marker_positions)
        if moved_markers:
            info_str = "; ".join([f"{m_id} -> {pos}" for (m_id, pos) in moved_markers])
            msg = f"[INFO] Updated markers. Moved: {info_str}"
        else:
            msg = "[INFO] No significant marker movement found."
        self.get_logger().info(msg)

        response.success = True
        response.message = msg
        return response

    # Example placeholder if you want to fetch by service:
    # def fetch_marker_callback(self, request, response):
    #     """ Suppose your custom service includes `request.marker_id` as input. """
    #     pos = fetch(request.marker_id)
    #     if pos:
    #         # fill response fields
    #         response.found = True
    #         response.position = list(pos)  # or however your custom service expects it
    #         response.message = f"Marker {request.marker_id} is at {pos}"
    #     else:
    #         response.found = False
    #         response.position = []
    #         response.message = f"Marker {request.marker_id} not found."
    #     return response

    def destroy_node(self):
        # Stop the RealSense pipeline on shutdown
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
