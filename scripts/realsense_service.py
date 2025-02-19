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
import threading
import time 
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from external_module import label_video
import pyrealsense2 as rs 
import cv2
import numpy as np


class realsense_service(Node):
    def __init__(self):
        super().__init__("realsense_service")
        self.get_logger().info("Realsense Video server Node intitialized")
        self.srv=self.create_service(SetBool, "/realsense_camera_control",self.camera_callback)
        self.pipeline=None
        self.video_writer=None
        self.capture_thread=None
        self.saving=False
        self.video_filename="realsense_video.avi"
    def capture_frames(self):
        try:
            while self.saving:
                frames=self.pipeline.wait_for_frames(timeout=1000)
                if not frames:
                    self.get_logger().error("No frames received from Realsense camera!")
                    self.saving=False
                    break

                color_frame=frames.get_color_frame()
                if not color_frame:
                    self.get_logger().error("No color frame available!")
                    self.saving=False
                    break

                frame=np.asanyarray(color_frame.get_data())
                self.video_writer.write(frame)
        except Exception as e:
            self.get_logger().error("Error in capture_frames:" + str(e))
            self.saving=False

    def camera_callback(self, req, resp):
        """
        Service callback.
        - If req.data is True (i.e. 1): start the RealSense camera and video capture.
        - If req.data is False (i.e. 0): stop video capture, shut down the camera,
          label the saved video, and return.
        The service returns 0 if starting the camera fails (e.g., no frames), 1 otherwise.
        """
        if req.data:  # Start capturing
            if self.saving:
                self.get_logger().warn("Camera is already running!")
                resp.success = True
                resp.message = "Camera already running (return code 1)"
                return resp

            self.get_logger().info("Starting RealSense camera and video capture")
            try:
                # Configure and start the RealSense pipeline
                self.pipeline = rs.pipeline()
                config = rs.config()
                # Enable the color stream at 640x480, 30 FPS
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(config)
            except Exception as e:
                self.get_logger().error("Failed to start RealSense pipeline: " + str(e))
                resp.success = False
                resp.message = "Camera start failed (return code 0)"
                return resp

            # Try to get one frame to verify the camera is sending video frames
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if not frames or not frames.get_color_frame():
                    self.get_logger().error("No video frames received from camera")
                    self.pipeline.stop()
                    self.pipeline = None
                    resp.success = False
                    resp.message = "No video frames (return code 0)"
                    return resp
            except Exception as e:
                self.get_logger().error("Error waiting for frames: " + str(e))
                self.pipeline.stop()
                self.pipeline = None
                resp.success = False
                resp.message = "No video frames (return code 0)"
                return resp

            # Setup video writer using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 30.0, (640, 480))
            self.saving = True

            # Start a thread to capture frames continuously
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.start()

            resp.success = True
            resp.message = "Camera started (return code 1)"
            return resp

        else:  # Stop capturing
            if not self.saving:
                self.get_logger().warn("Camera is not running!")
                resp.success = True
                resp.message = "Camera not running (return code 1)"
                return resp

            self.get_logger().info("Stopping video capture and shutting down camera")
            self.saving = False

            # Wait for the capture thread to finish
            if self.capture_thread is not None:
                self.capture_thread.join()
                self.capture_thread = None

            # Stop the RealSense pipeline and release the video writer
            try:
                self.pipeline.stop()
            except Exception as e:
                self.get_logger().error("Error stopping pipeline: " + str(e))
            self.pipeline = None

            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            # Label the video using the external function
            try:
                label_video(self.video_filename)
                self.get_logger().info("Video labeling complete")
            except Exception as e:
                self.get_logger().error("Error labeling video: " + str(e))

            resp.success = True
            resp.message = "Camera stopped and video labeled (return code 1)"
            return resp


def main():
    rclpy.init(args=None)
    node= realsense_service()
    try:
        rclpy.spin(node)
    except:
        node.get_logger().info("Shutting down Realsense Video server")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()