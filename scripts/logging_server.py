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
import sys, os
from ras_transport.interfaces.TransportWrapper import TransportFileClient

class LoggingServer(Node):
    def __init__(self):
        super().__init__('logging_server')
        self.topic_name = '/aruco_markers'
        self.image_topic = '/camera/camera/color/image_raw'
        self.depth_topic = '/camera/camera/depth/image_rect_raw'
        self.qos_value = 10
        self.timeout_value = 5
        self.bridge = CvBridge()
        self.logger = RasLogger()
        self.srv = self.create_service(StatusLog, '/traj_status', self.status_log_callback)
        self.status_log_client = self.create_client(StatusLog, '/send_logging')
        self.logger.log_info('Logging server started (step execution mode) using RasLogger')
        self.logger.log_info('Logging server initialized - will log data after each step execution')
        self.internal_step_counter = 0
        # Initialize file client for image upload
        self.file_client = TransportFileClient()
        # self.test_image_logging()

    def send_image(self, image_path: str, remote_filename: str) -> bool:
        """
        Upload a local image file to the file server via TransportFileClient.
        Args:
            image_path (str): Local path to the image file to upload.
            remote_filename (str): Filename to use on the server.
        Returns:
            bool: True if upload succeeded, False otherwise.
        """
        try:
            result = self.file_client.upload(image_path, remote_filename)
            if result:
                self.logger.log_info(f"Image uploaded to file server as {remote_filename}")
                return True
            else:
                self.logger.log_error(f"Failed to upload image as {remote_filename}")
                return False
        except Exception as e:
            self.logger.log_error(f"Exception during image upload: {e}")
            return False

    # def test_image_logging(self):
    #     """Test function to verify image logging works properly"""
    #     self.logger.log_info('Running image logging test')
        
    #     try:
    #         # Create a small test image (1x1 pixel PNG, but we'll convert to JPG for logging)
    #         test_image = np.zeros((1, 1, 3), dtype=np.uint8)
    #         test_image[0, 0] = [255, 255, 255]  # White pixel
    #         success_test, test_image_bytes = cv2.imencode('.jpg', test_image)
    #         self.logger.log_info("Starting image logging test")
    #         self.logger.log_json({"test": "image_logging", "status": "running"}, "Test metadata")
    #         if success_test:
    #             description = "Test tiny JPG image"
    #             desc_for_filename = "_" + description.replace(' ', '_').replace('/', '_')
    #             from datetime import datetime
    #             ts = self.logger._now() if hasattr(self.logger, '_now') else datetime.now().strftime('%Y%m%d_%H%M%S')
    #             img_filename = f"img_{ts}{desc_for_filename}.jpg"
    #             img_path = os.path.join(self.logger.base_log_dir if hasattr(self.logger, 'base_log_dir') else "logs", "images", img_filename)
    #             self.logger.log_image(test_image_bytes.tobytes(), description, "jpg")
    #             self.logger.log_info("Test tiny JPG image logged successfully")
    #             # Try uploading the test image to file server
    #             upload_success = self.send_image(img_path, img_filename)
    #             if upload_success:
    #                 self.logger.log_info(f"Test image {img_filename} uploaded to file server.")
    #             else:
    #                 self.logger.log_error(f"Failed to upload test image {img_filename} to file server.")
            
    #         # Also create and log a small color test image
    #         # Create a small 50x50 color test image with a red square
    #         color_test = np.zeros((50, 50, 3), dtype=np.uint8)
    #         color_test[10:40, 10:40] = [0, 0, 255]  # Red square in BGR format
            
    #         # Encode the test image to bytes
    #         success, color_bytes = cv2.imencode('.jpg', color_test)
    #         if success:
    #             self.logger.log_image(color_bytes.tobytes(), "Test color image with red square", "jpg")
    #             self.logger.log_info("Test color image logged")
            
    #         self.logger.log_info('Image logging test completed')
    #     except Exception as e:
    #         self.logger.log_error(f'Image logging test failed: {e}')

    def status_log_callback(self, request, response):
        """Called when a robot step is executed"""
        # Increment internal step counter
        self.internal_step_counter += 1
        step_num = self.internal_step_counter
        status = request.traj_status
        gripper_state = "closed" if request.gripper_status else "open"
        
        self.logger.log_info(f'Robot step {step_num} (request step: {request.current_traj}) executed with status: {status}, gripper: {gripper_state}')
        
        # Log trajectory status
        log_data = {
            "step": step_num,
            "request_step": request.current_traj,
            "status": status,
            "gripper": gripper_state
        }
        self.logger.log_json(log_data, description=f"Step {step_num} execution")
        self.logger.log_info(f"Processing step {step_num} (request step: {request.current_traj}) with status {status}")
        
        # Get the camera image
        cv_image = self.get_camera_images()
        if cv_image is None:
            self.logger.log_error("Could not get camera image")
            response.success = False
            return response
        
        # Process and log data
        timestamp = time.time()
        
        # 1. Log raw sensor data (color image only)
        # log_raw_sensor_data now returns the image filename used for upload
        img_filename = self.log_raw_sensor_data(step_num, cv_image, timestamp)
        
        # 2. Log perception data from ArUco markers
        self.log_perception_data(step_num)

        # 3. After logging/uploading the image, send the filename to the log sender via StatusLog service
        # import time
        start_time = time.time()
        if not self.status_log_client.wait_for_service(timeout_sec=2.0):
            self.logger.log_error('StatusLog service (log sender) not available!')
        else:
            from ras_interfaces.srv import StatusLog
            status_log_req = StatusLog.Request()
            status_log_req.traj_status = request.traj_status
            status_log_req.gripper_status = request.gripper_status
            status_log_req.current_traj = request.current_traj
            status_log_req.image_filename = img_filename
            future = self.status_log_client.call_async(status_log_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            status_log_resp = future.result()
            if status_log_resp and status_log_resp.success:
                self.logger.log_info(f"StatusLog sent to log sender with image filename {img_filename}.")
            else:
                self.logger.log_error("Failed to send StatusLog to log sender or no response.")
        elapsed = time.time() - start_time
        self.logger.log_info(f"StatusLog client call took {elapsed:.3f} seconds.")

        response.success = True
        return response
    
    def get_camera_images(self):
        """Get only the color image from camera"""
        # Get color image
        ret_img, img_msg = wait_for_message(Image, self, self.image_topic, time_to_wait=self.timeout_value)
        if not ret_img:
            self.logger.log_error(f'No image received on {self.image_topic}')
            return None
        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            return cv_image
        except Exception as e:
            self.logger.log_error(f"Image conversion error: {e}")
            return None
    
    def log_raw_sensor_data(self, step_num, cv_image, timestamp):
        """Log only the color camera image using RasLogger (JPG only) and upload to file server. Returns the image filename used for upload, or None on failure."""
        try:
            # Log color image as JPG
            success, img_bytes = cv2.imencode('.jpg', cv_image)
            if success:
                # Use RasLogger to log image (handles writing to disk)
                description = f"Color image for step {step_num}"
                self.logger.log_image(img_bytes.tobytes(), description, ext="jpg")

                # Reconstruct the filename and path as RasLogger does
                from datetime import datetime
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                desc_for_filename = "_" + description.replace(' ', '_').replace('/', '_')
                img_filename = f"img_{ts}{desc_for_filename}.jpg"
                img_path = os.path.join("logs", "images", img_filename)

                # Upload image to file server using send_image
                upload_success = self.send_image(img_path, img_filename)
                if upload_success:
                    self.logger.log_info(f"Image {img_filename} uploaded to file server.")
                else:
                    self.logger.log_error(f"Failed to upload {img_filename} to file server.")
                # Return the image filename regardless of upload success (so it can be included in StatusLog)
                return img_filename

            # Log metadata about raw sensor (only color image)
            raw_sensor_metadata = {
                "step": step_num,
                "timestamp": timestamp,
                "color_image_resolution": f"{cv_image.shape[1]}x{cv_image.shape[0]}"
            }
            self.logger.log_json(raw_sensor_metadata, description=f"Raw sensor metadata for step {step_num}")
            return None
        except Exception as e:
            self.logger.log_error(f"Error logging raw sensor data: {e}")
            return None
    
    def log_perception_data(self, step_num):
        """Log ArUco marker data"""
        # Log ArucoMarkers as JSON
        ret, msg = wait_for_message(ArucoMarkers, self, self.topic_name, qos_profile=self.qos_value, time_to_wait=self.timeout_value)
        if not ret:
            self.logger.log_error(f'No ArucoMarkers message received on {self.topic_name} for step {step_num}')
            return
        
        self.logger.log_info(f'Received ArucoMarkers on {self.topic_name} for step {step_num}')
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
