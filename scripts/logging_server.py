#!/usr/bin/env python3

"""
Copyright (C) 2024 Harsh Davda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For inquiries or further information, you may contact:
Harsh Davda
Email: info@opensciencestack.org
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from rclpy.wait_for_message import wait_for_message
from aruco_interfaces.msg import ArucoMarkers
from rclpy_message_converter import json_message_converter
from ras_interfaces.srv import StatusLog
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ras_logging.ras_logger import RasLogger

class LoggingServer(Node):
    
    def __init__(self):
        super().__init__('logging_server')
        self.logging_server = self.create_service(Empty, 'logging_server', self.logging_callback)
        self.topic_name = '/aruco_markers'
        self.image_topic = '/camera/camera/color/image_raw'
        self.qos_value = 10
        self.timeout_value = 3
        self.log_client = self.create_client(StatusLog, '/traj_status')
        self.bridge = CvBridge()
        self.logger = RasLogger()
        self.get_logger().info('Logging server started')

    def logging_callback(self, request, response):
        # Log ArucoMarkers as JSON
        ret, msg = wait_for_message(ArucoMarkers, self, self.topic_name, qos_profile=self.qos_value, time_to_wait=self.timeout_value)
        if not ret:
            self.get_logger().error('No message received for topic %s' % self.topic_name)
            self.logger.log_error(f'No ArucoMarkers message received on {self.topic_name}')
            return response
        else:
            self.get_logger().info('Message received for topic %s' % self.topic_name)
            req = StatusLog.Request()
            req.traj_status = "SUCCESS"
            req.gripper_status = False
            req.current_traj = 0
            self.log_client.call_async(req)
            json_msg = json_message_converter.convert_ros_message_to_json(msg)
            self.logger.log_json(json_msg, description="ArucoMarkers message")

        # Log a realsense screenshot (color image)
        ret_img, img_msg = wait_for_message(Image, self, self.image_topic, qos_profile=self.qos_value, time_to_wait=self.timeout_value)
        if not ret_img:
            self.get_logger().error(f'No image received on {self.image_topic}')
            self.logger.log_error(f'No image received on {self.image_topic}')
        else:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
                # Encode as JPEG
                success, img_bytes = cv2.imencode('.jpg', cv_image)
                if success:
                    self.logger.log_image(img_bytes.tobytes(), description="Realsense screenshot")
                    self.get_logger().info(f"Logged realsense screenshot from {self.image_topic}")
                else:
                    self.get_logger().error("Failed to encode image to JPEG")
                    self.logger.log_error("Failed to encode image to JPEG")
            except Exception as e:
                self.get_logger().error(f"Image conversion error: {e}")
                self.logger.log_error("Image conversion error", e)

        self.get_logger().info('Logging request received')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LoggingServer()
    while rclpy.ok():
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()