#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from aruco_interfaces.msg import ArucoMarkers
import tf2_ros
import tf2_geometry_msgs  # Needed for transforming PoseStamped
import traceback

class MoveToArucoMarker(Node):
    def __init__(self):
        super().__init__('move_to_aruco_marker')

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to ArUco marker topic
        self.subscription = self.create_subscription(
            ArucoMarkers,
            '/aruco_markers',
            self.aruco_callback,
            10
        )

        # Initialize MoveIt Action Client (C++ interface)
        from rclpy.action import ActionClient
        from moveit_msgs.action import MoveGroup
        self.moveit_action_client = ActionClient(self, MoveGroup, '/move_group')
        self.group_name = 'lite6'  # Use correct group name for Lite6
        self.end_effector_link = 'link_eef'  # Use correct end effector link
        self.target_frame = 'link_base'  # Planning frame (changed from 'base_link')
        self.get_logger().info("MoveToArucoMarker node initialized with MoveIt action client.")

    def aruco_callback(self, msg):
        if not msg.poses:
            return
        marker_pose = PoseStamped()
        marker_pose.header = msg.header
        marker_pose.pose = msg.poses[0]
        try:
            self.get_logger().info(f"Attempting transform from {marker_pose.header.frame_id} to {self.target_frame}")
            self.get_logger().info(f"Marker pose: {marker_pose.pose}")
            self.get_logger().info(f"marker_pose type: {type(marker_pose)}")
            self.get_logger().info(f"marker_pose.pose type: {type(marker_pose.pose)}")
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                marker_pose.header.frame_id,
                rclpy.time.Time())
            transformed_pose = tf2_geometry_msgs.do_transform_pose(marker_pose, trans)
            self.get_logger().info(f"Transformed pose: {transformed_pose.pose}")
            self.get_logger().info(f"transformed_pose type: {type(transformed_pose)}")
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}\n{traceback.format_exc()}")
            return
        # Optionally offset the marker pose (e.g., move above it)
        transformed_pose.pose.position.z += 0.1  # Lift 10 cm above

        # Prepare MoveIt goal
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, WorkspaceParameters
        from builtin_interfaces.msg import Duration

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = self.group_name
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.2
        goal_msg.request.max_acceleration_scaling_factor = 0.2
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.goal_constraints = [
            Constraints(
                position_constraints=[
                    PositionConstraint(
                        header=transformed_pose.header,
                        link_name=self.end_effector_link,
                        constraint_region=None  # None means use pose below
                    )
                ],
                orientation_constraints=[
                    OrientationConstraint(
                        header=transformed_pose.header,
                        orientation=transformed_pose.pose.orientation,
                        link_name=self.end_effector_link,
                        absolute_x_axis_tolerance=0.05,
                        absolute_y_axis_tolerance=0.05,
                        absolute_z_axis_tolerance=0.05,
                        weight=1.0
                    )
                ]
            )
        ]
        from geometry_msgs.msg import Vector3
        goal_msg.request.goal_constraints[0].position_constraints[0].target_point_offset = Vector3()
        goal_msg.request.start_state.is_diff = True
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 2
        goal_msg.planning_options.replan_delay = 2.0

        # Send goal to MoveIt
        if not self.moveit_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("MoveIt action server not available!")
            return
        send_goal_future = self.moveit_action_client.send_goal_async(goal_msg)

        def done_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("MoveIt did not accept goal.")
                return
            self.get_logger().info("MoveIt goal accepted, waiting for result...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_cb)

        def result_cb(future):
            result = future.result().result
            if result.error_code.val == 1:
                self.get_logger().info("MoveIt successfully planned and executed to ArUco marker!")
            else:
                self.get_logger().warn(f"MoveIt failed to plan: error code {result.error_code.val}")

        send_goal_future.add_done_callback(done_cb)

        self.get_logger().info(f"Sent target pose to MoveIt: {transformed_pose.pose}")

def main(args=None):
    rclpy.init(args=args)
    node = MoveToArucoMarker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
