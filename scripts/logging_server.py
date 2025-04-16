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
import os
import json
import time
from ament_index_python.packages import get_package_share_directory
import numpy as np

class LoggingServer(Node):
    
    def __init__(self):
        super().__init__('logging_server')

        self.topic_name = '/aruco_markers'
        self.image_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.qos_value = 10
        self.timeout_value = 5
        self.bridge = CvBridge()
        
        # Initialize RasLogger - this handles both local and remote logging
        self.logger = RasLogger()
        
        # Create service for trajectory step execution logging
        self.srv = self.create_service(StatusLog, '/traj_status', self.status_log_callback)
        
        self.get_logger().info('Logging server started (step execution mode) using RasLogger')
        self.logger.log_info('Logging server initialized - will log data after each step execution')
        
        # Run test image logging on startup
        self.test_image_logging()

    def test_image_logging(self):
        """Test function to verify image logging works properly"""
        self.get_logger().info('Running image logging test')
        
        try:
            # Create a small test image (1x1 pixel PNG)
            test_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82'
            
            # Log some test messages
            self.logger.log_info("Starting image logging test")
            self.logger.log_json({"test": "image_logging", "status": "running"}, "Test metadata")
            
            # Log the test image
            self.logger.log_image(test_image_bytes, "Test tiny PNG image", "png")
            self.logger.log_info("Test image logged successfully")
            
            # Also create and log a small color test image
            # Create a small 50x50 color test image with a red square
            color_test = np.zeros((50, 50, 3), dtype=np.uint8)
            color_test[10:40, 10:40] = [0, 0, 255]  # Red square in BGR format
            
            # Encode the test image to bytes
            success, color_bytes = cv2.imencode('.jpg', color_test)
            if success:
                self.logger.log_image(color_bytes.tobytes(), "Test color image with red square", "jpg")
                self.get_logger().info("Test color image logged")
            
            self.get_logger().info('Image logging test completed')
        except Exception as e:
            self.get_logger().error(f'Image logging test failed: {e}')
            self.logger.log_error(f'Image logging test failed: {e}')

    def status_log_callback(self, request, response):
        """Called when a robot step is executed"""
        step_num = request.current_traj
        status = request.traj_status
        gripper_state = "closed" if request.gripper_status else "open"
        
        self.get_logger().info(f'Robot step {step_num} executed with status: {status}, gripper: {gripper_state}')
        
        # Log trajectory status
        log_data = {
            "step": step_num,
            "status": status,
            "gripper": gripper_state
        }
        self.logger.log_json(log_data, description=f"Step {step_num} execution")
        self.logger.log_info(f"Processing step {step_num} with status {status}")
        
        # Get the camera images
        cv_image, depth_image = self.get_camera_images()
        if cv_image is None:
            self.get_logger().error("Could not get camera images")
            response.success = False
            return response
        
        # Process and log data
        timestamp = time.time()
        
        # 1. Log raw sensor data (camera images)
        self.log_raw_sensor_data(step_num, cv_image, depth_image, timestamp)
        
        # 2. Log perception data from ArUco markers
        self.log_perception_data(step_num)
        
        response.success = True
        return response
    
    def get_camera_images(self):
        """Get color and depth images from camera"""
        # Get color image
        ret_img, img_msg = wait_for_message(Image, self, self.image_topic, time_to_wait=self.timeout_value)
        if not ret_img:
            self.get_logger().error(f'No image received on {self.image_topic}')
            self.logger.log_error(f'No image received on {self.image_topic}')
            return None, None
            
        # Get depth image
        ret_depth, depth_msg = wait_for_message(Image, self, self.depth_topic, time_to_wait=self.timeout_value)
        if not ret_depth:
            self.get_logger().error(f'No depth image received on {self.depth_topic}')
            self.logger.log_error(f'No depth image received on {self.depth_topic}')
            return None, None
            
        try:
            # Convert images
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            return cv_image, depth_image
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            self.logger.log_error(f"Image conversion error: {e}")
            return None, None
    
    def log_raw_sensor_data(self, step_num, cv_image, depth_image, timestamp):
        """Log raw camera images using RasLogger"""
        try:
            # Log color image
            success, img_bytes = cv2.imencode('.jpg', cv_image)
            if success:
                self.logger.log_image(img_bytes.tobytes(), description=f"Color image for step {step_num}")
                self.get_logger().info(f"Logged color image for step {step_num}")
            
            # Log depth image (normalized for visualization)
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            success_depth, depth_bytes = cv2.imencode('.png', depth_normalized)
            if success_depth:
                self.logger.log_image(depth_bytes.tobytes(), description=f"Depth image for step {step_num}", ext="png")
                self.get_logger().info(f"Logged depth image for step {step_num}")
            
            # Log metadata about raw sensors
            raw_sensor_metadata = {
                "step": step_num,
                "timestamp": timestamp,
                "color_image_resolution": f"{cv_image.shape[1]}x{cv_image.shape[0]}",
                "depth_image_resolution": f"{depth_image.shape[1]}x{depth_image.shape[0]}"
            }
            
            self.logger.log_json(raw_sensor_metadata, description=f"Raw sensor metadata for step {step_num}")
            
            return True
        except Exception as e:
            self.get_logger().error(f"Error logging raw sensor data: {e}")
            self.logger.log_error(f"Error logging raw sensor data: {e}")
            return False
    
    def log_perception_data(self, step_num):
        """Log ArUco marker data"""
        # Log ArucoMarkers as JSON
        ret, msg = wait_for_message(ArucoMarkers, self, self.topic_name, qos_profile=self.qos_value, time_to_wait=self.timeout_value)
        if not ret:
            self.get_logger().error(f'No ArucoMarkers message received on {self.topic_name}')
            self.logger.log_error(f'No ArucoMarkers message received on {self.topic_name} for step {step_num}')
            return
        
        self.get_logger().info(f'Received ArucoMarkers on {self.topic_name} for step {step_num}')
        json_msg = json_message_converter.convert_ros_message_to_json(msg)
        
        # Add step number to the message for better tracking
        if isinstance(json_msg, dict):
            json_msg['step_num'] = step_num
        
        self.logger.log_json(json_msg, description=f"ArucoMarkers for step {step_num}")

def main(args=None):
    rclpy.init(args=args)
    node = LoggingServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()