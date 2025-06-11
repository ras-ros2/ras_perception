#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from tf2_ros import TransformBroadcaster
import yaml
import os
import ast
from ras_common.globals import RAS_CONFIGS_PATH

class TransformPublisher(Node):
    def __init__(self):
        super().__init__('transform_publisher')

        # Create a TransformBroadcaster instance to send transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Load transform values from file
        self.transform_values = self._load_transform_values()

        if self.transform_values:
            # Setup timer to broadcast transforms at regular intervals
            self.timer = self.create_timer(0.1, self.publish_transform)
        else:
            self.get_logger().error("Failed to load transform values. Node will not publish transforms.")

    def _load_transform_values(self):
        """Loads transform values from the static calibration file."""
        # Paths
        lab_setup_path = os.path.join(RAS_CONFIGS_PATH, 'lab_setup.yaml')
        static_calib_dir = os.path.join(RAS_CONFIGS_PATH, 'static_calibration_configs')
        
        # Read lab_setup.yaml
        try:
            with open(lab_setup_path, 'r') as f:
                lab_setup = yaml.safe_load(f)
            self.get_logger().info(f"Loaded lab_setup.yaml: {lab_setup}")
            if not lab_setup or 'lab_setup' not in lab_setup:
                raise KeyError("Key 'lab_setup' not found in lab_setup.yaml")
            if 'static_calibration_file' not in lab_setup['lab_setup']:
                raise KeyError("Key 'static_calibration_file' not found in lab_setup.yaml['lab_setup']")
            static_calib_file = lab_setup['lab_setup']['static_calibration_file']
        except Exception as e:
            self.get_logger().error(f'Error reading lab_setup.yaml or static_calibration_file: {e}')
            return None

        static_calib_path = os.path.join(static_calib_dir, static_calib_file)
        if not os.path.exists(static_calib_path):
            self.get_logger().error(f'Static calibration file not found: {static_calib_path}')
            return None

        # Parse file for transform arguments
        try:
            with open(static_calib_path, 'r') as f:
                content = f.read()
            
            # Extract the arguments list
            start = content.index('arguments=[') + len('arguments=[')
            end = content.index(']', start)
            args_str = content[start:end]
            
            # Convert to python list
            args = ast.literal_eval('[' + args_str + ']')
            
            # Map argument flags to values
            arg_map = {}
            i = 0
            while i < len(args):
                if str(args[i]).startswith('--'):
                    key = str(args[i]).lstrip('--')
                    value = args[i+1]
                    # Only convert to float for known numeric fields
                    if key in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']:
                        value = float(value)
                    arg_map[key] = value
                    i += 2
                else:
                    i += 1
            
            self.get_logger().info(f"Transform values read from calibration file: {arg_map}")
            return arg_map
        except Exception as e:
            self.get_logger().error(f'Error parsing static calibration file: {e}')
            return None

    def publish_transform(self):
        # Create TransformStamped message
        transform = geometry_msgs.msg.TransformStamped()

        # Set the header information for the transform
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'link_base'  # Parent frame
        transform.child_frame_id = 'camera_link'  # Child frame

        # Fill transform from loaded values
        transform.transform.translation.x = self.transform_values.get('x', 0.0)
        transform.transform.translation.y = self.transform_values.get('y', 0.0)
        transform.transform.translation.z = self.transform_values.get('z', 0.0)
        transform.transform.rotation.x = self.transform_values.get('qx', 0.0)
        transform.transform.rotation.y = self.transform_values.get('qy', 0.0)
        transform.transform.rotation.z = self.transform_values.get('qz', 0.0)
        transform.transform.rotation.w = self.transform_values.get('qw', 1.0)

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info('Published transform from link_base to camera_link') # Commented out to reduce log spam

def main(args=None):
    rclpy.init(args=args)

    # Create and spin the node to keep it alive
    node = TransformPublisher()
    rclpy.spin(node)

    # Shutdown after the node has finished
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()