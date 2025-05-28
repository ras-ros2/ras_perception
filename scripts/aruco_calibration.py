#!/usr/bin/env python3

"""
ArUco Calibration Script
This script detects ArUco markers with IDs 0-3 placed in four corners to form a rectangle.
It highlights the rectangle in light blue and calculates and highlights the center point.
"""

import rclpy
import sys
import cv2
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from cv2 import aruco
from geometry_msgs.msg import Point, Pose, TransformStamped
import tf2_ros
from aruco_interfaces.msg import ArucoMarkers
from ras_logging.ras_logger import RasLogger
import os
import json
import datetime
from pathlib import Path
from ras_common.globals import RAS_CONFIGS_PATH

class ArucoCalibration(Node):
    """
    ROS2 node for ArUco marker-based calibration.
    Detects ArUco markers with IDs 0-3, forms a rectangle, and highlights it.
    """

    def __init__(self):
        super().__init__('aruco_calibration')
        
        # Initialize variables
        self.cv_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        self.logger = RasLogger()
        
        # Variables for controlling calibration data writing
        self.write_enabled = True  # Always write calibration data
        
        # Path for temporary calibration data file
        self.temp_calibration_file = os.path.join(RAS_CONFIGS_PATH, "temp_calibration.json")
        self.last_write_time = 0  # To control write frequency
        
        # Store the inverse transformation from link_base to camera_link
        self.inverse_transform = None
        
        # Store workspace markers in robot base frame
        self.workspace_markers_base_frame = {}
        self.workspace_center_base_frame = None
        
        # Create subscriptions for color and depth images from RealSense camera
        self.color_cam_sub = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_image_callback, 
            10
        )
        
        self.depth_cam_sub = self.create_subscription(
            Image, 
            '/camera/camera/depth/image_rect_raw', 
            self.depth_image_callback, 
            10
        )
        
        # Create publisher for the processed image (raw)
        self.processed_image_pub = self.create_publisher(
            Image,
            '/aruco_calibration/processed_image',
            10
        )
        
        # Create publisher for the compressed processed image
        self.compressed_image_pub = self.create_publisher(
            CompressedImage,
            '/aruco_calibration/processed_image/compressed',
            10
        )
        
        # Create publisher for the center point
        self.center_point_pub = self.create_publisher(
            Point,
            '/aruco_calibration/center_point',
            10
        )
        
        # Create publisher for calibration target position
        self.calibration_target_pub = self.create_publisher(
            ArucoMarkers,
            '/calibration_target',
            10
        )
        
        # Set up TF2 components
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        
        # Removed pick location highlighting functionality
        
        # Set up timer for processing images
        self.timer = self.create_timer(0.1, self.process_image)  # 10 Hz
        
        # Camera parameters (from aruco_detection.py)
        self.cam_mat = np.array([
            [931.1829833984375, 0.0, 640.0], 
            [0.0, 931.1829833984375, 360.0], 
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # ArUco dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 8
        
        # No MQTT functionality
        
        self.logger.log_info("ArUco Calibration node started")

    def color_image_callback(self, data):
        """Process the color image from RealSense camera"""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            self.logger.log_error(f'Error converting color image: {e}')

    def depth_image_callback(self, data):
        """Process the depth image from RealSense camera"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except Exception as e:
            self.logger.log_error(f'Error converting depth image: {e}')

    def detect_aruco_markers(self, image):
        """
        Detect ArUco markers in the given image
        Returns dictionary with marker IDs as keys and corner coordinates as values
        """
        if image is None:
            return {}
        
        # Convert to grayscale for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        brightness = 2
        contrast = 1
        enhanced_image = cv2.addWeighted(gray, contrast, np.zeros(gray.shape, gray.dtype), 0, brightness)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            enhanced_image, 
            self.dictionary,
            parameters=self.aruco_params
        )
        
        # Initialize markers dictionary
        markers = {}
        
        # If markers are detected
        if ids is not None:
            # Flatten the IDs array
            ids = ids.flatten()
            
            # Process each marker
            for i, marker_id in enumerate(ids):
                # Get the corners of the marker
                marker_corners = corners[i][0]
                
                # Calculate the center of the marker
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Store the center coordinates
                markers[marker_id] = (center_x, center_y)
                
                # Draw the marker outline and ID on the image
                cv2.aruco.drawDetectedMarkers(image, corners)
                
                # Draw the marker ID and center point
                cv2.putText(
                    image, 
                    f"ID: {marker_id}", 
                    (center_x, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255), 
                    2
                )
                cv2.circle(image, (center_x, center_y), 2, (255, 0, 0), 6)
        
        return markers

    # Removed load_experiment_file method - no longer needed since pick location highlighting has been removed
    
    def write_calibration_data_to_file(self, x, y, z):
        """Write calibration data to a temporary file for IoT receiver to read"""
        try:
            # Get current time with nanoseconds for more precise timing
            current_time_msg = self.get_clock().now().to_msg()
            current_time = current_time_msg.sec + (current_time_msg.nanosec / 1e9)
            
            # Limit write frequency to once every 0.5 seconds to ensure more frequent updates
            if current_time - self.last_write_time < 0.5:
                return
            
            self.last_write_time = current_time
            
            # Convert to centimeters with 2 decimal places
            data = {
                "x": round(x * 100, 2),  # Convert to cm with 2 decimal places
                "y": round(y * 100, 2),  # Convert to cm with 2 decimal places
                "z": round(z * 100, 2),  # Convert to cm with 2 decimal places
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Write the data to the file
            with open(self.temp_calibration_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.get_logger().info(f"Calibration data written to file: X={data['x']}cm, Y={data['y']}cm, Z={data['z']}cm")
        except Exception as e:
            self.get_logger().error(f"Error writing calibration data to file: {e}")
    
    def process_image(self):
        """Process the image to detect markers and draw the rectangle"""
        if self.cv_image is None or self.depth_image is None:
            return
        
        # Make a copy of the image to draw on
        self.output_image = self.cv_image.copy()
        
        # Detect ArUco markers
        markers = self.detect_aruco_markers(self.output_image)
        
        # Check if we have all four markers (IDs 0-3)
        if len(markers) == 4 and all(i in markers for i in range(4)):
            # Get the corners of the rectangle
            corners = [markers[i] for i in range(4)]
            
            # Draw the rectangle connecting the markers
            # Use light blue color (255, 255, 0) with thickness 2
            for i in range(4):
                cv2.line(
                    self.output_image, 
                    corners[i], 
                    corners[(i + 1) % 4], 
                    (255, 255, 0), 
                    2
                )
            
            # Calculate the center of the rectangle
            center_x = sum(x for x, _ in corners) // 4
            center_y = sum(y for _, y in corners) // 4
            
            # Draw the center point with a red circle
            cv2.circle(self.output_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(
                self.output_image, 
                "TARGET", 
                (center_x + 10, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            
            # Draw X and Y axis arrows to show coordinate system
            arrow_length = 80  # Length of the arrows in pixels
            arrow_thickness = 2
            # X-axis arrow (vertical in image, pointing up) - RED
            cv2.arrowedLine(self.output_image, 
                           (center_x, center_y), 
                           (center_x, center_y - arrow_length), 
                           (0, 0, 255),  # Red color
                           arrow_thickness, 
                           tipLength=0.3)
            # Y-axis arrow (horizontal in image, pointing LEFT) - RED
            cv2.arrowedLine(self.output_image, 
                           (center_x, center_y), 
                           (center_x - arrow_length, center_y), 
                           (0, 0, 255),  # Red color
                           arrow_thickness, 
                           tipLength=0.3)
            # Add +X and +Y labels (positioned to avoid overlapping with arrows)
            cv2.putText(self.output_image, "+X", (center_x + 10, center_y - arrow_length//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(self.output_image, "+Y", (center_x - arrow_length - 20, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Get depth at center point
            try:
                depth = float(self.depth_image[center_y, center_x])
            except Exception as e:
                self.get_logger().error(f'Error getting depth at center: {e}')
                depth = 0.0
                
            # Camera parameters (from aruco_detection.py)
            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 640 
            centerCamY = 360
            focalX = 931.1829833984375
            focalY = 931.1829833984375
            
            # Calculate 3D position in camera frame (in meters)
            # For RealSense camera: Z is forward (depth), X is right, Y is down
            # Using the correct formula from the provided code
            z_cam = depth/1000.0  # Convert to meters (depth is in mm)
            x_cam = (depth * (sizeCamX - center_x - centerCamX) / focalX)/1000.0
            y_cam = (depth * (sizeCamY - center_y - centerCamY) / focalY)/1000.0
            
            self.logger.log_info(f'Camera frame coordinates: x={z_cam:.3f}m, y={x_cam:.3f}m, z={y_cam:.3f}m')
            
            # Publish the center point in image coordinates
            center_point = Point()
            center_point.x = float(center_x)
            center_point.y = float(center_y)
            center_point.z = depth
            self.center_point_pub.publish(center_point)
            
            # Log the center point
            self.logger.log_info(
                f'Rectangle center: ({center_x}, {center_y}), Depth: {depth}'
            )
            
            # Create transform for the center point in camera frame
            transform_msg = TransformStamped()
            transform_msg.header.stamp = self.get_clock().now().to_msg()
            transform_msg.header.frame_id = 'camera_link'
            transform_msg.child_frame_id = 'calibration_target'
            
            # Set the correct coordinate mapping for RealSense camera
            # Following the same approach as in the provided code
            transform_msg.transform.translation.x = float(z_cam)  # Depth is along X axis in base frame
            transform_msg.transform.translation.y = float(x_cam)  # Y in base frame
            transform_msg.transform.translation.z = float(y_cam)  # Z in base frame
            
            # Use identity rotation for the target
            transform_msg.transform.rotation.x = 0.0
            transform_msg.transform.rotation.y = 0.0
            transform_msg.transform.rotation.z = 0.0
            transform_msg.transform.rotation.w = 1.0
            
            # Broadcast the transform
            self.br.sendTransform(transform_msg)
            
            # Create ArucoMarkers message for the center point
            aruco_markers_msg = ArucoMarkers()
            aruco_markers_msg.header.stamp = self.get_clock().now().to_msg()
            aruco_markers_msg.marker_ids = [999]  # Use a special ID for the calibration target
            
            # Calculate and store workspace markers if we have all 4 markers (IDs 0-3)
            workspace_markers = {}
            for marker_id, marker_pos in markers.items():
                if marker_id in [0, 1, 2, 3]:
                    # We'll calculate and store these in base frame later
                    workspace_markers[marker_id] = marker_pos
            
            # Try to transform the point to link_base frame
            try:
                t = self.tf_buffer.lookup_transform(
                    'link_base', 
                    transform_msg.child_frame_id, 
                    rclpy.time.Time()
                )
                
                # Create a pose for the target in link_base frame
                pose = Pose()
                
                # Apply the transformation from tf lookup
                raw_x = t.transform.translation.x
                raw_y = t.transform.translation.y
                raw_z = t.transform.translation.z
                
                # Log the raw transformed values for debugging
                self.logger.log_info(f'Raw transform values: x={raw_x:.3f}m, y={raw_y:.3f}m, z={raw_z:.3f}m')
                
                # Set the final position values
                pose.position.x = raw_x
                pose.position.y = raw_y
                pose.position.z = raw_z
                
                # Store the workspace center coordinates for display
                # This follows the approach from the provided code
                if len(workspace_markers) == 4:
                    self.logger.log_info(f"Found all 4 workspace markers: {list(workspace_markers.keys())}")
                    self.logger.log_info(f"Workspace center at: x={raw_x*100:.1f}cm, y={raw_y*100:.1f}cm, z={raw_z*100:.1f}cm")
                    
                    # Print formatted output to console for easier viewing
                    print(f"\033[1;32m[WORKSPACE CENTER] Robot base frame (cm): X={raw_x*100:.1f}, Y={raw_y*100:.1f}, Z={raw_z*100:.1f}\033[0m")
                    
                    # Note: We'll write to file later in the code to avoid duplicate writes
                
                # Set orientation
                pose.orientation.x = t.transform.rotation.x
                pose.orientation.y = t.transform.rotation.y
                pose.orientation.z = t.transform.rotation.z
                pose.orientation.w = t.transform.rotation.w
                
                aruco_markers_msg.poses = [pose]
                
                # Log the transformed position (now in meters)
                self.logger.log_info(
                    f'Target in link_base frame: x={pose.position.x:.3f}m, '
                    f'y={pose.position.y:.3f}m, z={pose.position.z:.3f}m'
                )
                
                # Publish the target position to ROS topic
                self.calibration_target_pub.publish(aruco_markers_msg)
                
                # Always write calibration data to file regardless of whether all 4 markers were found
                # This ensures we keep writing even if we temporarily lose sight of some markers
                if self.write_enabled:
                    try:
                        # Write calibration data to file
                        self.write_calibration_data_to_file(raw_x, raw_y, raw_z)
                        self.logger.log_info("Wrote calibration target data to file")
                    except Exception as e:
                        self.logger.log_error(f"Error writing calibration target data to file: {e}")
                
                # Store the transform for future use if needed
                self.inverse_transform = t
                
                # Publish the processed image (raw and compressed)
                try:
                    # Publish raw image
                    img_msg = self.bridge.cv2_to_imgmsg(self.output_image, "bgr8")
                    self.processed_image_pub.publish(img_msg)
                    
                    # Publish compressed image
                    compressed_msg = CompressedImage()
                    compressed_msg.header = img_msg.header
                    compressed_msg.format = "jpeg"
                    # Encode the image as JPEG with quality 80
                    _, jpeg_img = cv2.imencode('.jpg', self.output_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    compressed_msg.data = np.array(jpeg_img).tobytes()
                    self.compressed_image_pub.publish(compressed_msg)
                    
                    self.logger.log_info("Published processed image (raw and compressed)")
                except Exception as e:
                    self.logger.log_error(f'Error publishing processed image: {e}')
                
            except Exception as e:
                self.logger.log_error(f'Error transforming to link_base: {e}')
                # If transformation fails, publish empty poses
                aruco_markers_msg.poses = []
                self.calibration_target_pub.publish(aruco_markers_msg)
        
        # Display the image
        cv2.imshow("ArUco Calibration", self.output_image)
        cv2.waitKey(1)
        
    def _get_timestamp(self):
        """Generate a timestamp string"""
        return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Removed draw_pick_locations method - pick location highlighting functionality has been removed

def main():
    rclpy.init(args=sys.argv)
    node = ArucoCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
