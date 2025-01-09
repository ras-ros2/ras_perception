#!/usr/bin/env python3

# ROS2 Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String

class InventoryNode(Node):
    def __init__(self):
        super().__init__("inventory_node")

        self.initialize_parameters()

        # Inventory structure
        self.inventory = {
            "white_markers": {},  # White marker poses
            "black_markers": {},  # Black marker poses
            "board": {            # Board state
                "cells": {},      # Dictionary of cells with their status and pose
                "center_pose": None,  # Center position of the board
                "total_empty_cells": 0  # Total number of empty cells
            }
        }

        # Subscribers
        self.perception_sub = self.create_subscription(PoseArray, self.perception_topic, self.perception_callback, 10)

        # Publishers
        self.inventory_pub = self.create_publisher(String, self.inventory_topic, 10)

        self.get_logger().info("Inventory Node Initialized.")

    def initialize_parameters(self):
        # Declare and retrieve parameters
        self.declare_parameter("perception_topic", "/detected_object_poses", "Pose topic from Perception Node")
        self.declare_parameter("inventory_topic", "/inventory/state", "Inventory publishing topic")
        self.declare_parameter("cell_size", 0.1, "Size of each cell in meters (square side length)")

        self.perception_topic = self.get_parameter("perception_topic").get_parameter_value().string_value
        self.inventory_topic = self.get_parameter("inventory_topic").get_parameter_value().string_value
        self.cell_size = self.get_parameter("cell_size").get_parameter_value().double_value

    def perception_callback(self, msg):
        # Update inventory with the detected poses
        self.update_markers(msg.poses)

        # Update board state based on marker positions
        self.update_board()

        # Publish the updated inventory
        self.publish_inventory()

    def update_markers(self, poses):
        # Clear existing marker data
        self.inventory["white_markers"].clear()
        self.inventory["black_markers"].clear()

        for i, pose in enumerate(poses):
            # Assign markers to white or black based on index (customize as needed)
            marker_type = "white_markers" if i < 5 else "black_markers"
            marker_id = len(self.inventory[marker_type]) + 1
            self.inventory[marker_type][marker_id] = pose

    def update_board(self):
        # Define the board geometry relative to the center pose
        center_pose = self.inventory["board"]["center_pose"]

        if center_pose is None:
            self.get_logger().warn("Board center pose not set! Skipping board update.")
            return

        # Calculate cell positions
        half_size = self.cell_size / 2.0
        offsets = [-self.cell_size, 0, self.cell_size]

        cell_index = 1
        empty_cells_count = 0

        for row_offset in offsets:
            for col_offset in offsets:
                # Calculate the position of each cell center
                pose = Pose()
                pose.position.x = center_pose.position.x + col_offset
                pose.position.y = center_pose.position.y + row_offset
                pose.position.z = center_pose.position.z

                # Determine if the cell is filled or empty
                cell_status = self.check_cell_status(pose)

                if cell_status == "empty":
                    empty_cells_count += 1

                # Update the inventory for this cell
                self.inventory["board"]["cells"][cell_index] = {
                    "pose": pose,
                    "status": cell_status
                }
                cell_index += 1

        # Record the total number of empty cells
        self.inventory["board"]["total_empty_cells"] = empty_cells_count

    def check_cell_status(self, cell_pose):
        """
        Checks if a cell is filled or empty by verifying if a marker is located inside it.
        """
        for marker_type in ["white_markers", "black_markers"]:
            for marker_pose in self.inventory[marker_type].values():
                if self.is_within_cell(cell_pose, marker_pose):
                    return "filled"

        return "empty"

    def is_within_cell(self, cell_pose, marker_pose):
        """
        Checks if a marker is within the bounds of a cell.
        """
        dx = abs(cell_pose.position.x - marker_pose.position.x)
        dy = abs(cell_pose.position.y - marker_pose.position.y)
        return dx <= self.cell_size / 2 and dy <= self.cell_size / 2

    def publish_inventory(self):
        # Convert inventory to a JSON-like string for simplicity
        inventory_state = str(self.inventory)
        msg = String()
        msg.data = inventory_state
        self.inventory_pub.publish(msg)
        self.get_logger().info(f"Published inventory state. Total empty cells: {self.inventory['board']['total_empty_cells']}")

def main():
    rclpy.init()
    node = InventoryNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
