#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from tf2_ros import TransformBroadcaster

class TransformPublisher(Node):
    def __init__(self):
        super().__init__('transform_publisher')

        # Create a TransformBroadcaster instance to send transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Setup timer to broadcast transforms at regular intervals
        self.timer = self.create_timer(0.1, self.publish_transform)

    def publish_transform(self):
        # Create TransformStamped message
        transform = geometry_msgs.msg.TransformStamped()

        # Set the header information for the transform
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'link_base'  # Parent frame
        transform.child_frame_id = 'camera_link'  # Child frame

        # Define the translation (camera_link ahead and slightly above link_base)
        transform.transform.translation.x = 0.68  # 1 meter ahead along the X axis
        transform.transform.translation.y = -0.16  # No displacement along Y axis
        transform.transform.translation.z = 0.97  # 0.5 meters above link_base

        # Define the rotation (camera's X axis should point downward)
        # To make the camera_link's X-axis point downward, we need to rotate it by 90 degrees around the X-axis.
        transform.transform.rotation.x = 0.0  # No rotation around X-axis
        transform.transform.rotation.y = -0.7071  # 90-degree rotation around Y-axis
        transform.transform.rotation.z = -0.0  # No rotation around Z-axis
        transform.transform.rotation.w = -0.7071  # 90-degree rotation around Y-axis

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info('Published transform from link_base to camera_link')

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