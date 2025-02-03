import pyrealsense2 as rs
import cv2
import numpy as np
import math
import os
import yaml
import time
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk  # To convert OpenCV images to Tkinter compatible

from plane_module import fit_plane_to_points_ransac
from camera_pose_module import (
    estimate_camera_pose_from_markers,
    compute_camera_pose
)
from intersection_module import intersect_ray_with_plane  # Not used currently
from yolo_module import (
    load_yolo_model,
    detect_objects,
    median_depth_in_bbox_center,
    compute_tictactoe_cell_centers_2d
)

# -------------------------------------------------------------------------
# Global settings / constants
# -------------------------------------------------------------------------
YAML_FILE = r"Downloads\markers.yaml"
DISTANCE_THRESHOLD = 0.04  # For movement detection in "update()"
MARKER_LENGTH      = 0.04   # ArUco marker size in meters
OCCLUSION_RATIO    = 0.8    # 4/5 => if a point is < 0.8 * plane_dist from camera, consider it an "object" above table

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

dist_coeffs = None  # If you have them, set them here.

# YOLO model path
YOLO_MODEL_PATH = r"C:\Users\saman\Downloads\tictactoe_model_updated.pt"

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_params = cv2.aruco.DetectorParameters()

# Initialize YOLO model once
model = load_yolo_model(YOLO_MODEL_PATH)

# -------------------------------------------------------------------------
# YAML helpers for inventory
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
# Inventory-like functions
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
    Implements a safe zone approach:
    - For each existing marker, checks if a detection exists within its safe zone.
    - If no detection is found within the safe zone, the marker is considered moved.
    - A new detection outside all safe zones is considered the new position of the moved marker.
    
    Args:
        marker_positions (dict): Dictionary with keys "black" and "white", each containing a list of positions.
    
    Returns:
        list of tuples: Each tuple contains (moved_marker_id, new_position)
    """
    data = _load_yaml()
    old_markers = data["markers"]

    new_white = marker_positions["white"]
    new_black = marker_positions["black"]

    # Separate new detections by type
    new_detections = {
        "white": new_white.copy(),
        "black": new_black.copy()
    }

    # To keep track of which detections have been matched
    matched_detections = {
        "white": [],
        "black": []
    }

    moved_markers = []

    # Iterate over old markers
    for m_id, coords in old_markers.items():
        marker_type = 'white' if m_id.startswith('w') else 'black'
        old_pos = np.array([coords["x"], coords["y"], coords["z"]])

        # Initialize as not moved
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
                break  # Assume one-to-one matching

        if not moved and new_pos is not None:
            # Update the marker's position
            data["markers"][m_id] = {
                "x": float(new_pos[0]),
                "y": float(new_pos[1]),
                "z": float(new_pos[2])
            }
        elif moved:
            # Marker is considered moved. Look for a new detection outside all safe zones.
            # Iterate over the same type detections to find a new unmatched detection
            for idx, det_pos in enumerate(new_detections[marker_type]):
                if idx in matched_detections[marker_type]:
                    continue  # Already matched
                det_pos_np = np.array(det_pos)
                # Check if this detection is outside all existing safe zones
                outside_all_safe_zones = True
                for other_m_id, other_coords in old_markers.items():
                    if other_m_id == m_id:
                        continue  # Current marker already checked
                    other_type = 'white' if other_m_id.startswith('w') else 'black'
                    other_pos = np.array([other_coords["x"], other_coords["y"], other_coords["z"]])
                    if np.linalg.norm(det_pos_np - other_pos) <= DISTANCE_THRESHOLD:
                        outside_all_safe_zones = False
                        break
                if outside_all_safe_zones:
                    # Found a new position for the moved marker
                    new_pos = det_pos
                    # Update the marker's position
                    data["markers"][m_id] = {
                        "x": float(new_pos[0]),
                        "y": float(new_pos[1]),
                        "z": float(new_pos[2])
                    }
                    # Mark this detection as matched
                    matched_detections[marker_type].append(idx)
                    # Record the movement
                    moved_markers.append((m_id, new_pos))
                    break  # Assume one movement per marker

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
# Single-pass detection that returns:
#   - marker_positions = {"black": [...], "white": [...]}
#   - plane_normal, plane_point
#   - R_camera, T_camera
#   - corners_2d, ids, rvec, tvec
# -------------------------------------------------------------------------
def run_detection_once(color_image, depth_image, depth_scale):
    """
    Process a single frame: detect ArUco markers, detect objects using YOLO,
    compute 3D positions of detected markers, fit a plane to marker corners,
    and transform marker positions to the global coordinate frame.

    Parameters:
        color_image (numpy.ndarray): The color image from the camera.
        depth_image (numpy.ndarray): The depth image from the camera.
        depth_scale (float): The scaling factor to convert depth units to meters.

    Returns:
        tuple: (
            marker_positions (dict): {"black": [...], "white": [...]},
            plane_normal (numpy.ndarray): Normal vector of the fitted plane,
            plane_point (numpy.ndarray): A point on the fitted plane,
            R_camera (numpy.ndarray): Rotation matrix of the camera pose,
            T_camera (numpy.ndarray): Translation vector of the camera pose,
            corners_2d (list): List of detected ArUco marker corners,
            ids (numpy.ndarray): Array of detected ArUco marker IDs,
            rvec (numpy.ndarray): Rotation vector from solvePnP,
            tvec (numpy.ndarray): Translation vector from solvePnP
        )
    """
    # --------------------------
    # ArUco Detection
    # --------------------------
    corners_2d, ids, _ = cv2.aruco.detectMarkers(
        color_image, aruco_dict, parameters=aruco_params
    )
    rvec, tvec = None, None

    marker_positions = {"black": [], "white": []}

    # List to store points with associated tag IDs for plane fitting
    points_with_tags = []

    # We will do solvePnP using only the marker with the smallest ID.
    if ids is not None and len(ids) > 0:
        # 1) Choose the marker with the smallest ID
        min_id_index = np.argmin(ids)  # index of the smallest ID
        corners_min = corners_2d[min_id_index][0]  # shape(4,2)
        min_tag_id = ids[min_id_index][0]

        # 2) The 3D corners in that marker's local coordinate system
        marker_3d = get_aruco_3d_corners(MARKER_LENGTH)

        # Build small lists for solvePnP
        single_objpoints = []
        single_imgpoints = []
        for j in range(4):
            single_objpoints.append(marker_3d[j])
            single_imgpoints.append(corners_min[j])

            # Also, store each corner with its tag ID for plane fitting
            # We'll assume all corners belong to the same tag (min_tag_id)
            # This will be overridden later for all markers
            # Alternatively, you can associate each corner with its respective tag
            # Here, since we're iterating through the smallest ID marker, it's already handled
            # Actual 3D points from all markers will be handled below

        single_objpoints = np.array(single_objpoints, dtype=np.float32)
        single_imgpoints = np.array(single_imgpoints, dtype=np.float32)

        # solvePnP to estimate camera pose relative to the marker
        rvec, tvec = estimate_camera_pose_from_markers(
            single_imgpoints,
            single_objpoints,
            camera_matrix,
            dist_coeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    # Initialize plane + camera transforms
    plane_normal = np.array([0, 0, 1], dtype=float)
    plane_point = np.zeros(3, dtype=float)
    R_camera = np.eye(3, dtype=float)
    T_camera = np.zeros(3, dtype=float)

    # If we found at least one marker and solved the pose
    if rvec is not None and tvec is not None:
        R_camera, T_camera = compute_camera_pose(rvec, tvec)
        T_camera = T_camera.flatten()

        # --------------------------
        # Collect 3D Points with Tag IDs for Plane Fitting
        # --------------------------
        for i_m in range(len(ids)):
            tag_id = ids[i_m][0]  # Extract tag ID
            c2d = corners_2d[i_m][0]  # shape (4,2)
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

        # --------------------------
        # Fit Plane Using Modified RANSAC
        # --------------------------
        if len(points_with_tags) >= 3:
            plane_normal, plane_point = fit_plane_to_points_ransac(points_with_tags)
        else:
            plane_normal = np.array([0, 0, 1], dtype=float)
            plane_point = np.array([0, 0, 0], dtype=float)

    # --------------------------
    # YOLO Object Detection (Black and White Markers)
    # --------------------------
    boxes = detect_objects(model, color_image, conf=0.5)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls = int(box.cls[0])
        if cls not in [0, 2]:
            continue  # Only interested in black (0) and white (2) markers

        u_center = int((x1 + x2) / 2)
        v_center = int((y1 + y2) / 2)

        # Ray-plane intersection in camera coordinates
        intersection_cam = intersect_ray_with_plane(
            u_center, v_center,
            plane_normal, plane_point,
            FX, FY, CX, CY
        )
        if intersection_cam is None:
            continue  # Intersection failed

        # Flatten to avoid shape issues
        intersection_cam = intersection_cam.flatten()

        # Transform to global coordinates (the chosen marker's frame)
        intersection_global = R_camera.T @ intersection_cam + T_camera

        if cls == 0:
            # Black marker
            marker_positions["black"].append(tuple(intersection_global))
        else:
            # White marker
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
# GUI Application
# -------------------------------------------------------------------------
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print("[INFO] Depth Scale is:", self.depth_scale)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Load YOLO model
        # Already loaded globally as 'model'

        # Create a label to display the video
        self.video_label = tk.Label(window)
        self.video_label.pack()

        # Create buttons
        self.init_button = tk.Button(window, text="Initialize", command=self.initialize_markers)
        self.init_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.update_button = tk.Button(window, text="Update", command=self.update_markers_gui)
        self.update_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.fetch_button = tk.Button(window, text="Fetch", command=self.fetch_marker)
        self.fetch_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Label for Fetch result
        self.fetch_result_label = tk.Label(window, text="Marker Position: ")
        self.fetch_result_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Label for Grid Occupancy
        self.grid_status_label = tk.Label(window, text="Grid Occupancy: ")
        self.grid_status_label.pack(padx=10, pady=10)

        # Initialize variables
        self.running = True
        self.marker_positions = {"black": [], "white": []}

        # Start the video loop
        self.update_video()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update_video(self):
        if not self.running:
            return

        # Grab frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            self.window.after(10, self.update_video)
            return

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run detection on the current frame
        self.marker_positions, plane_normal, plane_point, R_camera, T_camera, corners_2d, ids, rvec, tvec = run_detection_once(
            color_image, depth_image, self.depth_scale
        )

        # Update the annotated frame
        annotated_image = color_image.copy()

        # Visualization of detected markers
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated_image, corners_2d)
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(
                    annotated_image,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    0.05
                )

        # Draw YOLO bounding boxes with labels
        boxes = detect_objects(model, color_image, conf=0.5)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])
            if cls not in [0, 2]:
                continue
            label = "Black" if cls == 0 else ("White" if cls == 2 else "Other")
            color_bbox = (0, 0, 255) if cls == 0 else ((255, 255, 255) if cls == 2 else (0, 255, 0))
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color_bbox, 2)
            cv2.putText(
                annotated_image, label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color_bbox, 2
            )

        # Convert the image to PIL format
        image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Update the image in the GUI
        self.video_label.configure(image=image)
        self.video_label.image = image

        # Schedule the next frame update
        self.window.after(10, self.update_video)

    def initialize_markers(self):
        if not self.running:
            return
        # Initialize markers using current detection
        if not self.marker_positions["black"] and not self.marker_positions["white"]:
            messagebox.showerror("Error", "No markers detected to initialize.")
            return

        initialize(self.marker_positions)
        messagebox.showinfo("Initialize", "Markers have been initialized and saved to markers.yaml.")

    def update_markers_gui(self):
        if not self.running:
            return
        # Update markers using current detection
        if not self.marker_positions["black"] and not self.marker_positions["white"]:
            messagebox.showerror("Error", "No markers detected to update.")
            return

        moved_markers = update_markers(self.marker_positions)
        if moved_markers:
            moved_info = "\n".join([f"Marker '{m_id}' moved to {tuple(pos)}" for m_id, pos in moved_markers])
            messagebox.showinfo("Update", f"Markers have been updated:\n{moved_info}")
        else:
            messagebox.showinfo("Update", "No significant marker movement found.")

        # After updating markers, compute grid occupancy
        grid_occupancy = self.compute_grid_occupancy()
        self.display_grid_occupancy(grid_occupancy)

    def compute_grid_occupancy(self):
        """
        Compute which cells of the Tic Tac Toe grid are occupied based on current marker positions.

        Returns:
            dict: Keys are cell labels (e.g., 'A1'), values are marker IDs (e.g., 'w1', 'b2') or '-' if empty.
        """
        # Define frame center
        frame_center_px = (CX, CY)

        # Compute cell centers in 2D
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

        # Convert 3D marker positions to 2D pixel positions
        # Fetch current marker positions from YAML
        data = _load_yaml()
        markers = data.get("markers", {})

        marker_pixel_positions = {}
        for m_id, coords in markers.items():
            x, y, z = coords["x"], coords["y"], coords["z"]
            # Project 3D point to 2D pixel coordinates
            # Assuming the camera coordinate system
            if z == 0:
                continue  # Avoid division by zero
            u = (x * FX) / z + CX
            v = (y * FY) / z + CY
            marker_pixel_positions[m_id] = (u, v)

        # Define pixel threshold for occupancy (adjust as needed)
        pixel_threshold = 50  # Example value, adjust based on actual grid size and camera setup

        # Assign markers to grid cells
        for label, center in zip(cell_labels, cell_centers_2d.flatten().reshape(-1, 2)):
            cell_u, cell_v = center
            for m_id, (marker_u, marker_v) in marker_pixel_positions.items():
                distance = math.hypot(marker_u - cell_u, marker_v - cell_v)
                if distance <= pixel_threshold:
                    grid_occupancy[label] = m_id
                    break  # Assign only the first marker within the threshold

        return grid_occupancy

    def display_grid_occupancy(self, grid_occupancy):
        """
        Display the occupancy status of the Tic Tac Toe grid on the GUI.

        Args:
            grid_occupancy (dict): Keys are cell labels (e.g., 'A1'), values are marker IDs (e.g., 'w1', 'b2') or '-' if empty.
        """
        occupancy_text = "Grid Occupancy:\n"
        for row in ['A', 'B', 'C']:
            for col in ['1', '2', '3']:
                label = f"{row}{col}"
                occupant = grid_occupancy.get(label, '-')
                occupancy_text += f"{label}: {occupant}\n"
        self.grid_status_label.config(text=occupancy_text)

    def fetch_marker(self):
        # Prompt user for marker ID
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
        self.pipeline.stop()
        self.window.destroy()

if __name__ == "__main__":
    App(tk.Tk(), "Tic Tac Toe Marker Detection and Inventory")
