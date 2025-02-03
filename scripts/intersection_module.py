# intersection_module.py

import numpy as np

def intersect_ray_with_plane(u, v,
                             plane_normal, plane_point,
                             fx, fy, cx, cy):
    """
    Cast a ray from camera center through pixel (u, v)
    and intersect with the plane defined by 'plane_normal' & 'plane_point'.

    plane_normal: unit normal to the plane
    plane_point: a point on the plane

    Returns:
      The intersection (x, y, z) in the camera frame, or None if parallel.
    """

    # Ray from camera center
    ray_dir = np.array([
        (u - cx) / fx,
        (v - cy) / fy,
        1.0
    ], dtype=float)

    ray_dir /= np.linalg.norm(ray_dir)

    # Plane eq: plane_normal . X + d = 0 => d = -plane_normal . plane_point
    d = -plane_normal.dot(plane_point)
    denom = plane_normal.dot(ray_dir)

    if abs(denom) < 1e-12:
        # Nearly parallel
        return None

    # Parameter t where intersection = t * ray_dir
    # plane_normal . (t*ray_dir) + d = 0 => t = - (d + 0) / (plane_normal . ray_dir)
    t = -(d) / denom

    # If t < 0, plane is "behind" the camera
    if t < 0:
        return None

    intersection = ray_dir * t
    return intersection
