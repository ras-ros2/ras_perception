#!/usr/bin/env python3

# ROS2 Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from ultralytics import YOLO
from cv_bridge import CvBridge
import numpy as np

class YoloPerceptionNode(Node):
    def __init__(self):
        super().__init__("yolo_perception_node")

        # Load parameters
        self.initialize_parameters()

        # Initialize YOLO model
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()
        self.intrinsic_matrix = None
        self.distortion_coefficients = None
        self.depth_frame = None

        # Subscribe to Camera Info topic
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_callback, qos_profile_sensor_data)

        # Synchronize RGB and Depth topics if depth input is enabled
        if self.use_depth_input:
            from message_filters import Subscriber, ApproximateTimeSynchronizer
            self.rgb_sub = Subscriber(self, Image, self.image_topic, qos_profile=qos_profile_sensor_data)
            self.depth_sub = Subscriber(self, Image, self.depth_image_topic, qos_profile=qos_profile_sensor_data)
            self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
            self.sync.registerCallback(self.rgb_depth_callback)
        else:
            self.image_sub = self.create_subscription(Image, self.image_topic, self.rgb_callback, qos_profile_sensor_data)

        # Publishers
        self.annotated_image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.poses_pub = self.create_publisher(PoseArray, self.poses_topic, 10)

        self.get_logger().info("YOLO Perception Node Initialized.")

    def initialize_parameters(self):
        self.declare_parameter("image_topic", "/camera/color/image_raw", ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description="RGB image topic."))
        self.declare_parameter("depth_image_topic", "/camera/aligned_depth_to_color/image_raw", ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description="Depth image topic."))
        self.declare_parameter("info_topic", "/camera/color/camera_info", ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description="Camera info topic."))
        self.declare_parameter("output_image_topic", "/yolo/annotated_image", ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description="Output image topic."))
        self.declare_parameter("poses_topic", "/detected_object_poses", ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description="Detected object poses topic."))
        self.declare_parameter("model_path", "tictactoe_model.pt", ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description="Path to the YOLO model."))
        self.declare_parameter("use_depth_input", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL, description="Use depth input for XYZ calculations."))

        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        self.info_topic = self.get_parameter("info_topic").get_parameter_value().string_value
        self.output_image_topic = self.get_parameter("output_image_topic").get_parameter_value().string_value
        self.poses_topic = self.get_parameter("poses_topic").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.use_depth_input = self.get_parameter("use_depth_input").get_parameter_value().bool_value

    def info_callback(self, msg):
        self.intrinsic_matrix = np.reshape(msg.k, (3, 3))
        self.distortion_coefficients = np.array(msg.d)
        self.get_logger().info("Camera info received.")
        self.destroy_subscription(self.info_sub)

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        if self.intrinsic_matrix is None:
            self.get_logger().warn("Camera info not received.")
            return

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        results = self.model.predict(rgb_image, conf=0.5)
        annotated_image = results[0].plot()

        poses = PoseArray()
        poses.header = Header()
        poses.header.stamp = rgb_msg.header.stamp
        poses.header.frame_id = rgb_msg.header.frame_id

        for box in results[0].boxes.data:
            x_min, y_min, x_max, y_max, conf, cls = box.cpu().numpy()
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            depth = depth_image[cy, cx] * 0.001  # Convert to meters

            if depth > 0:
                fx, fy = self.intrinsic_matrix[0, 0], self.intrinsic_matrix[1, 1]
                cx_intr, cy_intr = self.intrinsic_matrix[0, 2], self.intrinsic_matrix[1, 2]

                x = (cx - cx_intr) * depth / fx
                y = (cy - cy_intr) * depth / fy
                z = depth

                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                poses.poses.append(pose)

        self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))
        self.poses_pub.publish(poses)

    def rgb_callback(self, rgb_msg):
        if self.intrinsic_matrix is None:
            self.get_logger().warn("Camera info not received.")
            return

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        results = self.model.predict(rgb_image, conf=0.5)
        annotated_image = results[0].plot()

        self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))

def main():
    rclpy.init()
    node = YoloPerceptionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
