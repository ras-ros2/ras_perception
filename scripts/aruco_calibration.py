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
from sensor_msgs.msg import Image
from cv2 import aruco
from geometry_msgs.msg import Point, Pose, TransformStamped
import tf2_ros
from aruco_interfaces.msg import ArucoMarkers
from ras_logging.ras_logger import RasLogger
import yaml
import os
import json
import datetime
from pathlib import Path
from ras_common.globals import RAS_CONFIGS_PATH
from ras_transport.interfaces.TransportWrapper import TransportMQTTPublisher
from rclpy_message_converter import json_message_converter

# Import paho-mqtt for direct MQTT publishing
import paho.mqtt.client as mqtt

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
        
        # Variables for controlling calibration target publishing
        self.publish_start_time = None
        self.publishing_enabled = True
        self.publish_duration = 10.0  # Publish for 10 seconds
        
        # Load experiment file to get pick locations
        self.pick_locations = []
        self.load_experiment_file()
        
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
        
        # Create publisher for the processed image
        self.processed_image_pub = self.create_publisher(
            Image,
            '/aruco_calibration/processed_image',
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
        
        # Add a parameter for enabling/disabling pick location highlighting (true by default)
        self.declare_parameter('show_pick_locations', True)
        self.show_pick_locations = self.get_parameter('show_pick_locations').value
        self.logger.log_info(f"Pick location highlighting is {'enabled' if self.show_pick_locations else 'disabled'}")
        
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
        
        # Initialize MQTT publisher for calibration target data
        self.mqtt_pub = TransportMQTTPublisher("robot/calibration_target")
        try:
            self.mqtt_pub.connect_with_retries()
            # Get MQTT broker details
            if hasattr(self.mqtt_pub, 'client') and hasattr(self.mqtt_pub.client, '_host'):
                mqtt_host = self.mqtt_pub.client._host
                mqtt_port = self.mqtt_pub.client._port
                self.logger.log_info(f"Connected to MQTT broker at {mqtt_host}:{mqtt_port} for calibration target publishing")
            else:
                self.logger.log_info("Connected to MQTT broker for calibration target publishing (broker details not available)")
        except Exception as e:
            self.logger.log_error(f"Failed to connect to MQTT broker: {e}")
            
        # Initialize a direct MQTT client as a backup
        try:
            # Load MQTT configuration from ras_conf.yaml
            config_path = os.path.join(RAS_CONFIGS_PATH, 'ras_conf.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            mqtt_config = config.get('ras', {}).get('transport', {}).get('mqtt', {})
            mqtt_host = mqtt_config.get('ip', 'dev2.deepklarity.ai')
            # Ensure we're using the correct domain (.ai not .at)
            if mqtt_host == 'dev2.deepklarity.at':
                mqtt_host = 'dev2.deepklarity.ai'
            mqtt_port = int(mqtt_config.get('port', 9500))
            
            # Create a direct MQTT client
            self.direct_mqtt_client = mqtt.Client()
            self.logger.log_info(f"Connecting direct MQTT client to {mqtt_host}:{mqtt_port}")
            print(f"\033[1;32m[ROBOT] Connecting direct MQTT client to {mqtt_host}:{mqtt_port}\033[0m")
            self.direct_mqtt_client.connect(mqtt_host, mqtt_port)
            self.direct_mqtt_client.loop_start()  # Start the MQTT client loop
            self.logger.log_info(f"Connected direct MQTT client to {mqtt_host}:{mqtt_port}")
            print(f"\033[1;32m[ROBOT] Connected direct MQTT client to {mqtt_host}:{mqtt_port}\033[0m")
        except Exception as e:
            self.logger.log_error(f"Failed to connect direct MQTT client: {e}")
            self.direct_mqtt_client = None
        
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

    def load_experiment_file(self):
        """Load the experiment file to extract pick locations"""
        try:
            # First, read the calibration config file to get the experiment name
            calibration_config_path = os.path.join(RAS_CONFIGS_PATH, 'calibration_config.yaml')
            self.logger.log_info(f"Reading calibration config from: {calibration_config_path}")
            
            if not os.path.exists(calibration_config_path):
                self.logger.log_error(f"Calibration config file not found at {calibration_config_path}")
                return
                
            with open(calibration_config_path, 'r') as f:
                calibration_config = yaml.safe_load(f)
                
            if not calibration_config or 'experiment_name' not in calibration_config:
                self.logger.log_error("No experiment_name specified in calibration_config.yaml")
                return
                
            experiment_name = calibration_config['experiment_name']
            self.logger.log_info(f"Using experiment: {experiment_name}")
            
            # Now load the specified experiment file
            experiments_dir = os.path.join(RAS_CONFIGS_PATH, 'experiments')
            experiment_file = os.path.join(experiments_dir, f"{experiment_name}.yaml")
            
            if not os.path.exists(experiment_file):
                self.logger.log_error(f"Experiment file not found: {experiment_file}")
                return
                
            self.logger.log_info(f"Loading experiment file: {experiment_file}")
            
            with open(experiment_file, 'r') as f:
                experiment_data = yaml.safe_load(f)
            
            # Extract pick locations from the experiment file
            if 'Poses' in experiment_data:
                poses = experiment_data['Poses']
                for key, value in poses.items():
                    # Look for 'in' positions which are pick locations
                    if key.startswith('in') and key != 'in4' and key != 'in5' and key != 'in6':
                        self.pick_locations.append({
                            'name': key,
                            'x': value['x'],
                            'y': value['y'],
                            'z': value['z']
                        })
                
                self.logger.log_info(f"Found {len(self.pick_locations)} pick locations in experiment file")
                for loc in self.pick_locations:
                    self.logger.log_info(f"Pick location {loc['name']}: x={loc['x']}, y={loc['y']}, z={loc['z']}")
            else:
                self.logger.log_error("No 'Poses' section found in experiment file")
        
        except Exception as e:
            self.logger.log_error(f"Error loading experiment file: {e}")
    
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
                
                # Check if we should publish via MQTT (only for 10 seconds after first detection)
                current_time = self.get_clock().now()
                
                # If this is the first time we're detecting markers, start the timer
                if self.publish_start_time is None and self.publishing_enabled:
                    self.publish_start_time = current_time
                    self.logger.log_info(f"Starting calibration target publishing for {self.publish_duration} seconds")
                    print(f"\033[1;33m[CALIBRATION] Publishing calibration target data for {self.publish_duration} seconds\033[0m")
                
                # Check if we're still within the publishing window
                if self.publishing_enabled and self.publish_start_time is not None:
                    elapsed_seconds = (current_time - self.publish_start_time).nanoseconds / 1e9
                    
                    if elapsed_seconds <= self.publish_duration:
                        # Still within publishing window, publish to MQTT
                        try:
                            # Convert the ArucoMarkers message to JSON string, then parse back to dict
                            json_str = json_message_converter.convert_ros_message_to_json(aruco_markers_msg)
                            calibration_data = json.loads(json_str)
                            
                            # Add additional metadata
                            calibration_data['timestamp'] = self._get_timestamp()
                            calibration_data['type'] = 'calibration_target'
                            # Convert position to centimeters and round for easier reading
                            calibration_data['description'] = 'Workspace center coordinates'
                            calibration_data['workspace_center_cm'] = {
                                'x': round(raw_x * 100),
                                'y': round(raw_y * 100),
                                'z': round(raw_z * 100)
                            }
                            # Add time remaining information
                            time_remaining = self.publish_duration - elapsed_seconds
                            calibration_data['publishing_time_remaining'] = round(time_remaining, 1)
                            
                            # Publish to MQTT
                            self.mqtt_pub.publish(json.dumps(calibration_data))
                            self.logger.log_info(f"Published calibration target data to MQTT (time remaining: {time_remaining:.1f}s)")
                        except Exception as e:
                            self.logger.log_error(f"Error publishing calibration target data to MQTT: {e}")
                    else:
                        # Publishing window has expired
                        if self.publishing_enabled:  # Only log this once
                            self.publishing_enabled = False
                            self.logger.log_info("Calibration target publishing window has expired")
                            print(f"\033[1;33m[CALIBRATION] Stopped publishing calibration target data after {self.publish_duration} seconds\033[0m")
                else:
                    # Publishing is disabled
                    pass
                
                # Store the inverse transform for projecting pick locations
                self.inverse_transform = t
                
                # Project and draw pick locations if we have the transform (enabled by default)
                if self.inverse_transform is not None:
                    if self.show_pick_locations:
                        self.logger.log_info("Drawing pick locations...")
                        # self.draw_pick_locations(self.output_image)
                    else:
                        self.logger.log_info("Pick location highlighting is disabled")
                else:
                    self.logger.log_info("Cannot draw pick locations: inverse transform not available")
                
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
    
    def draw_pick_locations(self, image):
        """Draw pick locations from experiment file on the image"""
        try:
            # Check if we have the inverse transform
            if self.inverse_transform is None:
                self.logger.log_warning("Cannot draw pick locations: inverse transform not available")
                return
                
            # Log pick locations for debugging
            self.logger.log_info(f"Drawing {len(self.pick_locations)} pick locations")
            for loc in self.pick_locations:
                self.logger.log_info(f"Pick location {loc['name']}: x={loc['x']}, y={loc['y']}, z={loc['z']}")
            
            # Camera parameters (from aruco_detection.py)
            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 640 
            centerCamY = 360
            focalX = 931.1829833984375
            focalY = 931.1829833984375
            
            # For each pick location
            for i, location in enumerate(self.pick_locations):
                # Create a point in base frame
                # Check if the coordinates are already in meters or need conversion
                if location['x'] > 10:  # Likely in centimeters if > 10
                    base_x = location['x'] / 100.0  # Convert from cm to meters
                    base_y = location['y'] / 100.0
                    base_z = location['z'] / 100.0
                else:  # Already in meters
                    base_x = location['x']
                    base_y = location['y']
                    base_z = location['z']
                
                # Log the base frame coordinates
                self.logger.log_info(f"Pick location {i+1} in base frame: x={base_x:.3f}, y={base_y:.3f}, z={base_z:.3f}")
                
                # Get the transform from base to camera
                # The inverse_transform is from camera to base, so we need to invert it
                cam_x = base_x - self.inverse_transform.transform.translation.x
                cam_y = base_y - self.inverse_transform.transform.translation.y
                cam_z = base_z - self.inverse_transform.transform.translation.z
                
                # Rotate the point (simplified for now)
                # This is a basic approach - a full transformation would use quaternion math
                # For visualization purposes, this approximation should work
                
                # Project 3D point to 2D image coordinates
                # Using the standard pinhole camera model
                if cam_z > 0:  # Only project points in front of the camera
                    # Convert to camera coordinates
                    img_x = int(centerCamX + (cam_x / cam_z) * focalX)
                    img_y = int(centerCamY + (cam_y / cam_z) * focalY)
                    
                    # Log the image coordinates
                    self.logger.log_info(f"Projected pick location {i+1} to image coordinates: ({img_x}, {img_y})")
                    
                    # Check if the point is within the image bounds
                    if 0 <= img_x < sizeCamX and 0 <= img_y < sizeCamY:
                        # Draw a rectangle around the pick location
                        rect_size = 50  # Size of rectangle in pixels
                        color = (0, 255, 255)  # Yellow color for pick locations
                        cv2.rectangle(image, 
                                     (img_x - rect_size//2, img_y - rect_size//2),
                                     (img_x + rect_size//2, img_y + rect_size//2),
                                     color, 2)
                        
                        # Add a label
                        cv2.putText(image, 
                                   f"PLACE CUBE {i+1}", 
                                   (img_x - rect_size//2, img_y - rect_size//2 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Log successful drawing
                        self.logger.log_info(f"Successfully drew pick location {i+1} at ({img_x}, {img_y})")
                    else:
                        self.logger.log_warn(f"Pick location {i+1} is outside image bounds: ({img_x}, {img_y})")
                else:
                    self.logger.log_warn(f"Pick location {i+1} is behind the camera (z={cam_z})")
                
                # This line is now handled inside the if-else blocks above
                
        except Exception as e:
            self.logger.log_error(f"Error drawing pick locations: {e}")
        
        # Publish the processed image
        try:
            img_msg = self.bridge.cv2_to_imgmsg(self.output_image, "bgr8")
            self.processed_image_pub.publish(img_msg)
        except Exception as e:
            self.logger.log_error(f'Error publishing processed image: {e}')

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
