#!/usr/bin/env python3

import base64
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import roslibpy

class MultiCompressedImageBridgeNode(Node):
    def __init__(self):
        super().__init__('multi_compressed_image_bridge_node')

        # Subscribe to aruco_calibration processed image
        self.subscription_aruco = self.create_subscription(
            CompressedImage,
            '/aruco_calibration/processed_image/compressed',
            self.aruco_image_callback,
            10)

        # Subscribe to camera depth image
        self.subscription_depth = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.depth_image_callback,
            10)

        # Initialize rosbridge client
        self.rosbridge_client = roslibpy.Ros(host='localhost', port=9090)
        self.rosbridge_client.on_ready(self.on_rosbridge_ready)
        self.rosbridge_client.on('close', self.on_rosbridge_disconnect)
        self.rosbridge_client.on('error', self.on_rosbridge_error)

        self.publisher_aruco = None
        self.publisher_depth = None

    def on_rosbridge_ready(self):
        self.get_logger().info('Connected to rosbridge server')

        # Publisher for aruco processed image
        self.publisher_aruco = roslibpy.Topic(
            self.rosbridge_client,
            '/aruco_calibration/processed_image/compressed',
            'sensor_msgs/CompressedImage'
        )

        # Publisher for depth image
        self.publisher_depth = roslibpy.Topic(
            self.rosbridge_client,
            '/camera/camera/color/image_raw/compressed',
            'sensor_msgs/CompressedImage'
        )

    def on_rosbridge_disconnect(self, *args):
        self.get_logger().warn('Connection to rosbridge server disconnected')

    def on_rosbridge_error(self, error):
        self.get_logger().error(f'Rosbridge connection error: {error}')

    def aruco_image_callback(self, msg):
        if not self.rosbridge_client.is_connected or self.publisher_aruco is None:
            self.get_logger().warn('Rosbridge not connected or aruco publisher not ready, skipping publish')
            return

        encoded_data = base64.b64encode(msg.data).decode('utf-8')

        ros_msg = {
            'header': {
                'stamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                },
                'frame_id': msg.header.frame_id
            },
            'format': msg.format,
            'data': encoded_data
        }

        try:
            self.publisher_aruco.publish(ros_msg)
            self.get_logger().debug('Published aruco compressed image message')
        except Exception as e:
            self.get_logger().error(f'Failed to publish aruco image message: {e}')

    def depth_image_callback(self, msg):
        if not self.rosbridge_client.is_connected or self.publisher_depth is None:
            self.get_logger().warn('Rosbridge not connected or depth publisher not ready, skipping publish')
            return

        encoded_data = base64.b64encode(msg.data).decode('utf-8')

        ros_msg = {
            'header': {
                'stamp': {
                    'sec': msg.header.stamp.sec,
                    'nanosec': msg.header.stamp.nanosec
                },
                'frame_id': msg.header.frame_id
            },
            'format': msg.format,
            'data': encoded_data
        }

        try:
            self.publisher_depth.publish(ros_msg)
            self.get_logger().debug('Published depth compressed image message')
        except Exception as e:
            self.get_logger().error(f'Failed to publish depth image message: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiCompressedImageBridgeNode()

    ros2_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros2_thread.start()

    try:
        node.get_logger().info('Connecting to rosbridge server...')
        node.rosbridge_client.run_forever()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down...')
    finally:
        if node.publisher_aruco:
            node.publisher_aruco.unadvertise()
        if node.publisher_depth:
            node.publisher_depth.unadvertise()
        node.rosbridge_client.terminate()
        node.destroy_node()
        rclpy.shutdown()
        ros2_thread.join(timeout=2)

if __name__ == '__main__':
    main()
