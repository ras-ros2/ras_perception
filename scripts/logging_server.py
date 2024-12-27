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

class LoggingServer(Node):
    
    def __init__(self):
        super().__init__('logging_server')
        self.logging_server = self.create_service(Empty, 'logging_server', self.logging_callback)
        self.topic_name = '/aruco_markers'
        self.qos_value = 10
        self.timeout_value = 3
        self.get_logger().info('Logging server started')

    def logging_callback(self, request, response):
        ret, msg = wait_for_message(ArucoMarkers, self, self.topic_name, qos_profile=self.qos_value, time_to_wait=self.timeout_value)
        if not ret:
            self.get_logger().error('No message received for topic %s' % self.topic_name)
            return response
        else:
            self.get_logger().info('Message received for topic %s' % self.topic_name)
            json_msg = json_message_converter.convert_ros_message_to_json(msg)
            with open('pose_log2.json', 'a') as log_file:
                log_file.write(f'{json_msg}\n')

        self.get_logger().info('Logging request received')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LoggingServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()