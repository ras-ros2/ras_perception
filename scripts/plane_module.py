# plane_module.py

import numpy as np
import cv2
import time
import random

_LAST_LOG_TIME = 0.0
_LOG_INTERVAL = 10.0  # seconds

def fit_plane_to_points_ransac(points_with_tags, dist_threshold=0.002, max_iters=500):
    """
    Fit a plane to a set of 3D points using a RANSAC approach, ensuring that
    the sampled points come from different ArUco markers to improve robustness.

    Parameters:
        points_with_tags (list of tuples): Each tuple contains (tag_id, point),
                                           where tag_id is an identifier for the ArUco tag,
                                           and point is a numpy array of shape (3,).
        dist_threshold (float): Distance threshold in meters to determine inliers.
        max_iters (int): Maximum number of RANSAC iterations.

    Returns:
        tuple: (best_normal, best_point_on_plane)
               - best_normal (numpy array): Normal vector of the best-fit plane.
               - best_point_on_plane (numpy array): A point lying on the best-fit plane.
    """

    if len(points_with_tags) < 3:
        # Not enough points to define a plane
        return np.array([0, 0, 1], dtype=float), np.zeros(3, dtype=float)

    # Organize points by their tag IDs
    tag_to_points = {}
    for tag_id, point in points_with_tags:
        if tag_id not in tag_to_points:
            tag_to_points[tag_id] = []
        tag_to_points[tag_id].append(point)

    unique_tags = list(tag_to_points.keys())

    if len(unique_tags) < 3:
        # Need at least 3 distinct tags to sample points from different markers
        return np.array([0, 0, 1], dtype=float), np.zeros(3, dtype=float)

    best_inlier_count = -1
    best_normal = None
    best_point_on_plane = None

    # Precompute all possible unique triplets of tags to improve sampling efficiency
    # This is optional and can be commented out if not desired
    # from itertools import combinations
    # all_tag_triplets = list(combinations(unique_tags, 3))
    # random.shuffle(all_tag_triplets)

    for _ in range(max_iters):
        try:
            # 1. Randomly select 3 unique tag IDs
            sampled_tags = random.sample(unique_tags, 3)

            # 2. From each selected tag, randomly pick one point
            sampled_points = [random.choice(tag_to_points[tag]) for tag in sampled_tags]

            p1, p2, p3 = sampled_points

            # 3. Check for collinearity
            v1 = p2 - p1
            v2 = p3 - p1
            maybe_normal = np.cross(v1, v2)
            norm_val = np.linalg.norm(maybe_normal)
            if norm_val < 1e-9:
                continue  # Degenerate case; points are collinear or identical

            # 4. Normalize the normal vector
            maybe_normal /= norm_val

            # 5. Compute the plane distance from origin: n.x + d = 0 => d = -n.p1
            d = -np.dot(maybe_normal, p1)

            # 6. Compute distances of all points to the plane
            diffs = np.dot([pt for _, pt in points_with_tags], maybe_normal) + d
            dist = np.abs(diffs)
            inliers = dist < dist_threshold
            inlier_count = np.sum(inliers)

            # 7. Update the best plane if current one has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_normal = maybe_normal
                # Use the centroid of inliers as a representative point on the plane
                inlier_points = np.array([pt for (_, pt), is_inlier in zip(points_with_tags, inliers) if is_inlier])
                best_point_on_plane = np.mean(inlier_points, axis=0)

                # Early stopping if a plane with a high inlier ratio is found
                if inlier_count > 0.8 * len(points_with_tags):
                    break

        except ValueError as ve:
            # Handle cases where sampling fails due to insufficient tags
            print(f"[WARN] Sampling failed: {ve}")
            continue

    # Fallback if no good plane was found
    if best_normal is None:
        return np.array([0, 0, 1], dtype=float), np.zeros(3, dtype=float)

    # Ensure consistent orientation (e.g., Z component positive)
    if best_normal[2] < 0:
        best_normal = -best_normal

    # Low-frequency logging
    global _LAST_LOG_TIME
    current_time = time.time()
    if (current_time - _LAST_LOG_TIME) > _LOG_INTERVAL:
        print("[INFO] RANSAC plane fit log:")
        print(f"       Inliers = {best_inlier_count} / {len(points_with_tags)}")
        print(f"       Plane normal  = {best_normal}")
        print(f"       Plane centroid= {best_point_on_plane}")
        _LAST_LOG_TIME = current_time

    return best_normal, best_point_on_plane
