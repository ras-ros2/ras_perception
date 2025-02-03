# yolo_module.py

from ultralytics import YOLO
import numpy as np
import cv2
import math

def load_yolo_model(model_path):
    """
    Load the YOLO model from the given path.
    """
    model = YOLO(model_path)
    return model

def detect_objects(model, image, conf=0.5):
    """
    Run the model on the given image and return predictions.
    """
    results = model.predict(image, conf=conf)
    boxes = results[0].boxes  # YOLOv8's Boxes structure
    return boxes

def median_depth_in_bbox_center(depth_image, center_u, center_v, half_window=2):
    """
    Sample a (2*half_window+1) x (2*half_window+1) patch around (center_v, center_u)
    from the depth_image, then return the median valid depth (in raw depth units, not scaled).
    If all are invalid (0), returns 0.
    """
    h, w = depth_image.shape
    umin = max(center_u - half_window, 0)
    umax = min(center_u + half_window, w - 1)
    vmin = max(center_v - half_window, 0)
    vmax = min(center_v + half_window, h - 1)

    patch = depth_image[vmin:vmax+1, umin:umax+1].reshape(-1)
    patch_nonzero = patch[patch > 0]
    if len(patch_nonzero) == 0:
        return 0
    return np.median(patch_nonzero)

def compute_tictactoe_cell_centers_2d(frame_center_px,
                                      dist_edge_px=80,
                                      dist_corner_px=105,
                                      margin_scale=1.8):
    """
    Compute the 3x3 cell centers in 2D around the stored frame center.
    """
    cx_px, cy_px = frame_center_px
    corner_off_px = (dist_corner_px / np.sqrt(2)) * margin_scale
    dist_edge_px *= margin_scale

    cell_centers = np.zeros((3, 3, 2), dtype=float)
    for i, row in enumerate([-1, 0, 1]):
        for j, col in enumerate([-1, 0, 1]):
            if row == 0 and col == 0:
                x_off, y_off = (0.0, 0.0)
            elif row == 0:
                x_off = dist_edge_px * col
                y_off = 0.0
            elif col == 0:
                x_off = 0.0
                y_off = dist_edge_px * row
            else:
                x_off = corner_off_px * col
                y_off = corner_off_px * row

            cell_u = cx_px + x_off
            cell_v = cy_px + y_off
            cell_centers[i, j, 0] = cell_u
            cell_centers[i, j, 1] = cell_v

    return cell_centers
