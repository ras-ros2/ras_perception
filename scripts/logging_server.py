#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from aruco_interfaces.msg import ArucoMarkers
from rclpy_message_converter import json_message_converter
from ras_interfaces.srv import StatusLog
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ras_logging.ras_logger import RasLogger

class LoggingServer(Node):
    
    def __init__(self):
        super().__init__('logging_server')

        self.topic_name = '/aruco_markers'
        self.image_topic = '/camera/camera/color/image_raw'
        self.qos_value = 10
        self.timeout_value = 5
        self.bridge = CvBridge()
        self.logger = RasLogger()
        self.log_client = self.create_client(StatusLog, '/traj_status')

        # Start timer for automatic logging every 5 seconds
        self.create_timer(5.0, self.logging_callback)

        self.get_logger().info('Logging server started (automatic mode)')

    def logging_callback(self, _=None):
        # Log ArucoMarkers as JSON
        ret, msg = wait_for_message(ArucoMarkers, self, self.topic_name, qos_profile=self.qos_value, time_to_wait=self.timeout_value)
        if not ret:
            self.get_logger().error(f'No ArucoMarkers message received on {self.topic_name}')
            self.logger.log_error(f'No ArucoMarkers message received on {self.topic_name}')
        else:
            self.get_logger().info(f'Received ArucoMarkers on {self.topic_name}')
            req = StatusLog.Request()
            req.traj_status = "SUCCESS"
            req.gripper_status = False
            req.current_traj = 0
            self.log_client.call_async(req)

            json_msg = json_message_converter.convert_ros_message_to_json(msg)
            self.logger.log_json(json_msg, description="ArucoMarkers message")

        # Log RealSense screenshot
        ret_img, img_msg = wait_for_message(Image, self, self.image_topic, time_to_wait=self.timeout_value)
        if not ret_img:
            self.get_logger().error(f'No image received on {self.image_topic}')
            self.logger.log_error(f'No image received on {self.image_topic}')
        else:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
                success, img_bytes = cv2.imencode('.jpg', cv_image)
                if success:
                    self.logger.log_image(img_bytes.tobytes(), description="Realsense screenshot")
                    self.get_logger().info(f"Logged image from {self.image_topic}")
                else:
                    self.get_logger().error("Failed to encode image to JPEG")
                    self.logger.log_error("Failed to encode image to JPEG")
            except Exception as e:
                self.get_logger().error(f"Image conversion error: {e}")
                self.logger.log_error("Image conversion error", e)

def main(args=None):
    rclpy.init(args=args)
    node = LoggingServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()