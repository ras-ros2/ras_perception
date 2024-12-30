#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# Standard messages
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Pose, PoseStamped

# Custom ArUco detection message
# (Adjust the import to match your actual package/msg name)
from aruco_interfaces.msg import ArucoMarkers

# For transforms
import numpy as np
from tf_transformations import quaternion_matrix

# For RViz visualization
from visualization_msgs.msg import Marker, MarkerArray


class LatticeCenterCalculator(Node):
    def __init__(self):
        super().__init__('lattice_center_calculator')

        # Subscribe to the array of ArUco marker detections
        self.aruco_subscription = self.create_subscription(
            ArucoMarkers,
            '/aruco/markers',   # Adjust if different
            self.aruco_callback,
            10
        )

        # Publisher for the 5 lattice-section PoseStamped
        self.lattice_pub = self.create_publisher(
            PoseStamped,
            '/lattice_sections_poses',
            10
        )

        # Publisher for RViz visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/calculated_centers_markers',
            10
        )

        # The ID of the marker you want to track (the first section center)
        self.tracked_marker_id = 0

        # Distance (in meters) between section centers
        # (60 mm = 0.06 m, or adjust as measured)
        self.center_to_center_offset = 0.06

        # Number of sections in your lattice
        self.num_sections = 5

    def aruco_callback(self, msg: ArucoMarkers):
        """
        Called whenever we receive a list of detected markers.
        We'll look for the marker with 'tracked_marker_id' and
        compute the lattice centers from its pose.
        """
        # Basic validation
        if len(msg.marker_ids) != len(msg.poses):
            self.get_logger().error("Mismatch in marker_ids and poses array length!")
            return

        found_marker = False

        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == self.tracked_marker_id:
                found_marker = True
                marker_pose = msg.poses[i]

                # Convert geometry_msgs/Pose to PoseStamped
                marker_pose_stamped = PoseStamped()
                marker_pose_stamped.header = msg.header
                marker_pose_stamped.pose = marker_pose

                # Compute the center poses of all sections
                section_poses = self.compute_lattice_centers(marker_pose_stamped)

                # Publish each as a PoseStamped
                for idx, section_pose in enumerate(section_poses, start=1):
                    self.get_logger().info(
                        f"Section {idx} center: "
                        f"x={section_pose.pose.position.x:.3f}, "
                        f"y={section_pose.pose.position.y:.3f}, "
                        f"z={section_pose.pose.position.z:.3f}"
                    )
                    self.lattice_pub.publish(section_pose)

                # Also publish MarkerArray for RViz visualization
                marker_array = self.create_markers_for_rviz(section_poses)
                self.marker_pub.publish(marker_array)
                break  # Stop after finding our target marker

        if not found_marker:
            self.get_logger().warn(
                f"Marker ID {self.tracked_marker_id} not found in this frame."
            )

    def compute_lattice_centers(self, marker_pose_stamped: PoseStamped):
        """
        Given the PoseStamped of the marker (section 1 center),
        compute the PoseStamped for the entire lattice (5 sections).
        """
        pose = marker_pose_stamped.pose

        # Extract translation and quaternion
        position = np.array([pose.position.x,
                             pose.position.y,
                             pose.position.z])
        quaternion = np.array([pose.orientation.x,
                               pose.orientation.y,
                               pose.orientation.z,
                               pose.orientation.w])

        # Create a 4x4 homogeneous transform
        transform = quaternion_matrix(quaternion)
        transform[0:3, 3] = position

        # Build 5 PoseStamped (sections)
        section_pose_list = []

        for i in range(self.num_sections):
            x_offset = i * self.center_to_center_offset

            offset_vec = np.array([x_offset, 0.0, 0.0, 1.0]).reshape(4, 1)
            cam_vec = transform.dot(offset_vec)

            ps = PoseStamped()
            ps.header.frame_id = marker_pose_stamped.header.frame_id
            ps.header.stamp = self.get_clock().now().to_msg()

            ps.pose.position.x = cam_vec[0][0]
            ps.pose.position.y = cam_vec[1][0]
            ps.pose.position.z = cam_vec[2][0]

            # Keep the same orientation as the marker
            ps.pose.orientation = pose.orientation

            section_pose_list.append(ps)

        return section_pose_list

    def create_markers_for_rviz(self, pose_stamped_list):
        """
        Given a list of PoseStamped, create a MarkerArray (arrows)
        so you can visualize them (position + orientation) in RViz.
        """
        marker_array = MarkerArray()

        for idx, ps in enumerate(pose_stamped_list):
            marker = Marker()
            marker.header.frame_id = ps.header.frame_id
            marker.header.stamp = ps.header.stamp

            # Unique ID for each marker so RViz can track them
            marker.id = idx

            # We'll use an ARROW to show position + orientation
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Set the pose
            marker.pose = ps.pose

            # Set the scale (length, width, height of the arrow)
            # Adjust as needed to be visible at your scene scale
            marker.scale.x = 0.04  # length of arrow
            marker.scale.y = 0.004  # width of arrow
            marker.scale.z = 0.004  # height of arrow

            # Example color: semi-opaque blue
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.8

            # How long to keep the marker displayed
            # 0 = forever, or set a duration for debugging
            marker.lifetime.sec = 0

            marker_array.markers.append(marker)

        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = LatticeCenterCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
