#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import threading
import time
import cv2
import numpy as np
import traceback

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from ras_perception.msg import RealsenseData  # Adjust package name as necessary

import pyrealsense2 as rs
from cv_bridge import CvBridge

# -----------------------------------------------------------------------------
# Shared RealSense Pipeline (same as your existing code)
# -----------------------------------------------------------------------------
class SharedRealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print("[SharedRealSense] Failed to start pipeline:", e)
            raise
        self.align = rs.align(rs.stream.color)
        self.latest_frames = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if frames:
                    aligned_frames = self.align.process(frames)
                    with self.lock:
                        self.latest_frames = aligned_frames
            except Exception as e:
                print("[SharedRealSense] Exception in _update:", e)
                traceback.print_exc()
            time.sleep(0.01)

    def get_frames(self):
        with self.lock:
            return self.latest_frames

    def stop(self):
        self.running = False
        try:
            self.pipeline.stop()
        except Exception as e:
            print("[SharedRealSense] Exception during stop:", e)

shared_rs = SharedRealSense()  # Global instance

# -----------------------------------------------------------------------------
# Realsense Publisher Node
# -----------------------------------------------------------------------------
class RealsensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        self.publisher_ = self.create_publisher(RealsenseData, 'realsense_data', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()
        # Get the depth scale from the device.
        profile = shared_rs.pipeline.get_active_profile()
        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()
        self.get_logger().info(f"[INFO] Depth Scale: {self.depth_scale}")

    def timer_callback(self):
        frames = shared_rs.get_frames()
        if frames is None:
            return

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays.
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        try:
            # Convert images to ROS Image messages.
            rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="mono16")

            # Populate our custom message.
            msg = RealsenseData()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.rgb_image = rgb_msg
            msg.depth_image = depth_msg
            msg.depth_scale = float(self.depth_scale)

            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error("Failed to publish RealsenseData: " + str(e))
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    node = RealsensePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        shared_rs.stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
