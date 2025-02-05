#!/usr/bin/env python3
"""
GUI Node that subscribes to a ROS 2 topic (realsense_data) and replicates
the full functionality of the original code:
  - Uses incoming color and depth images to detect ArUco and YOLO markers.
  - Provides buttons to initialize (save to YAML), update, and fetch marker positions.
  - Computes and displays grid occupancy.

This version does NOT use multi-threading. Instead, the ROS 2 node is run
in the same thread as the Tkinter GUI by periodically calling rclpy.spin_once().
"""

import time
import math
import os
import yaml
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import traceback

import cv2
import numpy as np

# ROS 2 and cv_bridge imports
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# Import custom message (adjust package name as needed)
from ras_perception.msg import RealsenseData

# Import helper modules (ensure these are in your PYTHONPATH)
from plane_module import fit_plane_to_points_ransac
from camera_pose_module import estimate_camera_pose_from_markers, compute_camera_pose
from intersection_module import intersect_ray_with_plane
from yolo_module import load_yolo_model, detect_objects, median_depth_in_bbox_center, compute_tictactoe_cell_centers_2d

# -------------------------------------------------------------------------
# Global Settings / Constants
# -------------------------------------------------------------------------
YAML_FILE = r"markers.yaml"  # Adjust path if needed
DISTANCE_THRESHOLD = 0.04     # in meters (for safe-zone matching)
MARKER_LENGTH = 0.04          # ArUco marker size in meters
OCCLUSION_RATIO = 0.8         # (not used directly here)

# RealSense camera intrinsics (adjust these values to your camera)
FX = 921.1829
FY = 931.1829
CX = 640
CY = 360

camera_matrix = np.array([[FX, 0, CX],
                          [0, FY, CY],
                          [0,  0,  1]], dtype=np.float32)
dist_coeffs = None  # Set if available

# YOLO model path
YOLO_MODEL_PATH = r"/home/mdfh/ros2_ws/src/ras_perception/scripts/tictactoe_model_updated.pt"

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_params = cv2.aruco.DetectorParameters_create()

# Load YOLO model once
model = load_yolo_model(YOLO_MODEL_PATH)

# -------------------------------------------------------------------------
# YAML Helper Functions for Inventory
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
# Marker Helper: 3D Corners of an ArUco Marker
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
# Inventory Functions
# -------------------------------------------------------------------------
def initialize(marker_positions):
    """
    Save the current marker positions (for white and black markers) into a YAML file.
    White markers are labeled as w1, w2, … and black markers as b1, b2, …
    """
    data = {"markers": {}}
    white_list = marker_positions["white"]
    black_list = marker_positions["black"]

    w_count = 1
    b_count = 1
    for pos in white_list:
        marker_id = f"w{w_count}"
        data["markers"][marker_id] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
        w_count += 1
    for pos in black_list:
        marker_id = f"b{b_count}"
        data["markers"][marker_id] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
        b_count += 1

    _save_yaml(data)
    print("[INFO] Initialize: wrote markers to", YAML_FILE)

def update_markers(marker_positions):
    """
    Update marker positions using a safe-zone approach.
    Returns a list of tuples (marker_id, new_position) for markers that moved.
    """
    data = _load_yaml()
    old_markers = data["markers"]
    new_white = marker_positions["white"]
    new_black = marker_positions["black"]

    new_detections = {"white": new_white.copy(), "black": new_black.copy()}
    matched_detections = {"white": [], "black": []}
    moved_markers = []

    for m_id, coords in old_markers.items():
        marker_type = 'white' if m_id.startswith('w') else 'black'
        old_pos = np.array([coords["x"], coords["y"], coords["z"]])
        moved = True
        new_pos = None
        for idx, det_pos in enumerate(new_detections[marker_type]):
            det_pos_np = np.array(det_pos)
            if np.linalg.norm(det_pos_np - old_pos) <= DISTANCE_THRESHOLD:
                moved = False
                new_pos = det_pos
                matched_detections[marker_type].append(idx)
                break
        if not moved and new_pos is not None:
            data["markers"][m_id] = {"x": float(new_pos[0]), "y": float(new_pos[1]), "z": float(new_pos[2])}
        elif moved:
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
                    data["markers"][m_id] = {"x": float(new_pos[0]), "y": float(new_pos[1]), "z": float(new_pos[2])}
                    matched_detections[marker_type].append(idx)
                    moved_markers.append((m_id, new_pos))
                    break

    _save_yaml(data)
    if moved_markers:
        info = "\n".join([f"Marker '{m_id}' moved to {tuple(pos)}" for m_id, pos in moved_markers])
        print("[INFO] Updated markers:\n" + info)
    else:
        print("[INFO] No significant marker movement found.")
    return moved_markers

def fetch(marker_id):
    """
    Return the (x, y, z) position for the given marker from the YAML file.
    """
    data = _load_yaml()
    markers = data.get("markers", {})
    if marker_id not in markers:
        return None
    m = markers[marker_id]
    return (m["x"], m["y"], m["z"])

def compute_grid_occupancy():
    """
    Compute grid occupancy based on marker positions stored in YAML.
    Returns a dictionary mapping cell labels to marker IDs or '-' if empty.
    """
    frame_center_px = (CX, CY)
    cell_centers_2d = compute_tictactoe_cell_centers_2d(
        frame_center_px,
        dist_edge_px=80,
        dist_corner_px=105,
        margin_scale=1.8
    )
    cell_labels = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
    grid_occupancy = {label: '-' for label in cell_labels}
    data = _load_yaml()
    markers = data.get("markers", {})
    marker_pixel_positions = {}
    for m_id, coords in markers.items():
        x, y, z = coords["x"], coords["y"], coords["z"]
        if z == 0:
            continue
        u = (x * FX) / z + CX
        v = (y * FY) / z + CY
        marker_pixel_positions[m_id] = (u, v)
    pixel_threshold = 50
    cell_centers = cell_centers_2d.flatten().reshape(-1, 2)
    for label, center in zip(cell_labels, cell_centers):
        cell_u, cell_v = center
        for m_id, (marker_u, marker_v) in marker_pixel_positions.items():
            if math.hypot(marker_u - cell_u, marker_v - cell_v) <= pixel_threshold:
                grid_occupancy[label] = m_id
                break
    return grid_occupancy

# -------------------------------------------------------------------------
# Single-Pass Detection Function
# -------------------------------------------------------------------------
def run_detection_once(color_image, depth_image, depth_scale):
    """
    Process a single frame:
      - Detect ArUco markers.
      - Estimate camera pose using the marker with the smallest ID.
      - Compute 3D positions of marker corners using depth data.
      - Fit a plane to these points.
      - Run YOLO detection for markers (class 0: Black, class 2: White).
    Returns a tuple:
      (marker_positions, plane_normal, plane_point, R_camera, T_camera, corners_2d, ids, rvec, tvec)
    """
    # ArUco Detection
    corners_2d, ids, _ = cv2.aruco.detectMarkers(color_image, aruco_dict, parameters=aruco_params)
    rvec, tvec = None, None
    marker_positions = {"black": [], "white": []}
    points_with_tags = []

    if ids is not None and len(ids) > 0:
        min_id_index = np.argmin(ids)  # Choose marker with smallest ID
        corners_min = corners_2d[min_id_index][0]
        marker_3d = get_aruco_3d_corners(MARKER_LENGTH)
        single_objpoints = np.array(marker_3d, dtype=np.float32)
        single_imgpoints = np.array(corners_min, dtype=np.float32)
        rvec, tvec = estimate_camera_pose_from_markers(
            single_imgpoints,
            single_objpoints,
            camera_matrix,
            dist_coeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    # Default plane and camera transforms
    plane_normal = np.array([0, 0, 1], dtype=float)
    plane_point = np.zeros(3, dtype=float)
    R_camera = np.eye(3, dtype=float)
    T_camera = np.zeros(3, dtype=float)

    if rvec is not None and tvec is not None:
        R_camera, T_camera = compute_camera_pose(rvec, tvec)
        T_camera = T_camera.flatten()
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
        if len(points_with_tags) >= 3:
            plane_normal, plane_point = fit_plane_to_points_ransac(points_with_tags)
        else:
            plane_normal = np.array([0, 0, 1], dtype=float)
            plane_point = np.array([0, 0, 0], dtype=float)

    # YOLO Detection for markers (class 0: Black, class 2: White)
    boxes = detect_objects(model, color_image, conf=0.5)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        if cls not in [0, 2]:
            continue
        u_center = int((x1 + x2) / 2)
        v_center = int((y1 + y2) / 2)
        intersection_cam = intersect_ray_with_plane(u_center, v_center, plane_normal, plane_point, FX, FY, CX, CY)
        if intersection_cam is None:
            continue
        intersection_cam = intersection_cam.flatten()
        intersection_global = R_camera.T @ intersection_cam + T_camera
        if cls == 0:
            marker_positions["black"].append(tuple(intersection_global))
        else:
            marker_positions["white"].append(tuple(intersection_global))
    return (marker_positions, plane_normal, plane_point, R_camera, T_camera, corners_2d, ids, rvec, tvec)

# -------------------------------------------------------------------------
# ROS 2 Subscriber Node for RealSense Data
# -------------------------------------------------------------------------
class ROSImageSubscriber(Node):
    def __init__(self):
        super().__init__('gui_node_subscriber')
        self.subscription = self.create_subscription(
            RealsenseData,
            'realsense_data',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.depth_scale = None

    def listener_callback(self, msg: RealsenseData):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding="bgr8")
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding="mono16")
            self.depth_scale = msg.depth_scale
        except Exception as e:
            self.get_logger().error("Listener callback error: " + str(e))

# -------------------------------------------------------------------------
# Tkinter GUI Application using ROS Frames (Single Thread)
# -------------------------------------------------------------------------
class App:
    def __init__(self, window, window_title, ros_node):
        self.window = window
        self.window.title(window_title)
        self.ros_node = ros_node  # Use the ROS subscriber node

        # Video display label
        self.video_label = tk.Label(window)
        self.video_label.pack()

        # Buttons for Initialize, Update, and Fetch
        self.init_button = tk.Button(window, text="Initialize", command=self.initialize_markers)
        self.init_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.update_button = tk.Button(window, text="Update", command=self.update_markers_gui)
        self.update_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.fetch_button = tk.Button(window, text="Fetch", command=self.fetch_marker)
        self.fetch_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.fetch_result_label = tk.Label(window, text="Marker Position: ")
        self.fetch_result_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.grid_status_label = tk.Label(window, text="Grid Occupancy: ")
        self.grid_status_label.pack(padx=10, pady=10)

        self.running = True
        self.marker_positions = {"black": [], "white": []}

        # Start the video update loop (which will also process ROS messages)
        self.update_video()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update_video(self):
        if not self.running:
            return

        # Process any ROS messages (non-blocking)
        rclpy.spin_once(self.ros_node, timeout_sec=0.001)

        # Use frames from the ROS subscriber if available
        if (self.ros_node.latest_color is not None and
            self.ros_node.latest_depth is not None and
            self.ros_node.depth_scale is not None):
            color_image = self.ros_node.latest_color.copy()
            depth_image = self.ros_node.latest_depth.copy()
            depth_scale = self.ros_node.depth_scale

            # Run detection on the current frame
            (self.marker_positions, plane_normal, plane_point, R_camera, T_camera,
             corners_2d, ids, rvec, tvec) = run_detection_once(color_image, depth_image, depth_scale)

            # Annotate the image
            annotated_image = color_image.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(annotated_image, corners_2d)
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(annotated_image, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            boxes = detect_objects(model, color_image, conf=0.5)
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                if cls not in [0, 2]:
                    continue
                label = "Black" if cls == 0 else "White"
                color_bbox = (0, 0, 255) if cls == 0 else (255, 255, 255)
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color_bbox, 2)
                cv2.putText(annotated_image, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bbox, 2)

            # Convert image for Tkinter display
            image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.video_label.configure(image=image)
            self.video_label.image = image

        self.window.after(10, self.update_video)

    def initialize_markers(self):
        if not self.running:
            return
        if not self.marker_positions["black"] and not self.marker_positions["white"]:
            messagebox.showerror("Error", "No markers detected to initialize.")
            return
        # Use the current detection to initialize (save YAML)
        initialize(self.marker_positions)
        messagebox.showinfo("Initialize", "Markers have been initialized and saved to markers.yaml.")

    def update_markers_gui(self):
        if not self.running:
            return
        if not self.marker_positions["black"] and not self.marker_positions["white"]:
            messagebox.showerror("Error", "No markers detected to update.")
            return
        moved_markers = update_markers(self.marker_positions)
        if moved_markers:
            info_str = "\n".join([f"Marker '{m_id}' moved to {tuple(pos)}" for m_id, pos in moved_markers])
            messagebox.showinfo("Update", f"Markers have been updated:\n{info_str}")
        else:
            messagebox.showinfo("Update", "No significant marker movement found.")
        grid_occupancy = self.compute_grid_occupancy()
        self.display_grid_occupancy(grid_occupancy)

    def compute_grid_occupancy(self):
        frame_center_px = (CX, CY)
        cell_centers_2d = compute_tictactoe_cell_centers_2d(
            frame_center_px,
            dist_edge_px=80,
            dist_corner_px=105,
            margin_scale=1.8
        )
        cell_labels = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
        grid_occupancy = {label: '-' for label in cell_labels}

        data = _load_yaml()
        markers = data.get("markers", {})
        marker_pixel_positions = {}
        for m_id, coords in markers.items():
            x, y, z = coords["x"], coords["y"], coords["z"]
            if z == 0:
                continue
            u = (x * FX) / z + CX
            v = (y * FY) / z + CY
            marker_pixel_positions[m_id] = (u, v)

        pixel_threshold = 50
        cell_centers = cell_centers_2d.flatten().reshape(-1, 2)
        for label, center in zip(cell_labels, cell_centers):
            cell_u, cell_v = center
            for m_id, (marker_u, marker_v) in marker_pixel_positions.items():
                if math.hypot(marker_u - cell_u, marker_v - cell_v) <= pixel_threshold:
                    grid_occupancy[label] = m_id
                    break
        return grid_occupancy

    def display_grid_occupancy(self, grid_occupancy):
        occupancy_text = "Grid Occupancy:\n"
        for row in ['A', 'B', 'C']:
            for col in ['1', '2', '3']:
                label = f"{row}{col}"
                occupant = grid_occupancy.get(label, '-')
                occupancy_text += f"{label}: {occupant}\n"
        self.grid_status_label.config(text=occupancy_text)

    def fetch_marker(self):
        marker_id = simpledialog.askstring("Fetch Marker", "Enter Marker ID (e.g., w1, b2):")
        if marker_id:
            position = fetch(marker_id)
            if position:
                self.fetch_result_label.config(text=f"Marker Position: {position}")
                messagebox.showinfo("Fetch Marker", f"Marker {marker_id} position: {position}")
            else:
                self.fetch_result_label.config(text="Marker Position: Not found")
                messagebox.showwarning("Fetch Marker", f"Marker {marker_id} not found in {YAML_FILE}.")

    def on_closing(self):
        self.running = False
        rclpy.shutdown()  # Shutdown ROS before exiting
        self.window.destroy()

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    # Initialize ROS 2
    rclpy.init()
    # Create the ROS subscriber node (single-threaded)
    ros_node = ROSImageSubscriber()
    # Create and start the Tkinter GUI, passing in the ROS node
    App(tk.Tk(), "Tic Tac Toe Marker Detection and Inventory", ros_node)

if __name__ == "__main__":
    main()
