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
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage, Image
from cv2 import aruco
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
import tf_transformations
from aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Pose

def calculate_rectangle_area(coordinates):
		height = ((coordinates[0] - coordinates[4])**2 + (coordinates[1] - coordinates[5])**2)**0.5
		width = ((coordinates[2] - coordinates[4])**2 + (coordinates[3] - coordinates[5])**2)**0.5

		area = width * height

		return area, width 

def detect_aruco(image, depth):

		center_aruco_list = []
		distance_from_rgb_list = []
		angle_aruco_list = []
		width_aruco_list = []
		ids = []
		flat_list =[]
		aruco_area_threshold = 1500

		cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]],dtype=np.float32)

		dist_mat = np.array([0.0,0.0,0.0,0.0,0.0])

		size_of_aruco_cm = 15
		dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

		try:
			arucoParams = cv2.aruco.DetectorParameters()

			arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
			arucoParams.cornerRefinementWinSize = 8  

	
			brightness = 2
			contrast = 1
			image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)

			(corners, aruco_id, rejected) = cv2.aruco.detectMarkers(image, dictionary, parameters=arucoParams)
			for i in range(len(aruco_id)):

				x1= corners[i][0][0][0]
				y1= corners[i][0][0][1]
				x2= corners[i][0][2][0]
				y2= corners[i][0][2][1]
				x3= corners[i][0][3][0]
				y3= corners[i][0][3][1]  

				coordinates= [x1,y1,x2,y2,x3,y3]
				area, width = calculate_rectangle_area(coordinates)

				if area >= aruco_area_threshold:
					rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, size_of_aruco_cm, cam_mat, dist_mat)
					ids.append(int(aruco_id[i]))
					current_rvec = rvec[i]
					current_tvec = tvec[i]
					
					center_aruco = np.mean(corners[i][0], axis=0)
					center_x, center_y = map(int, center_aruco)
					distance_from_rgb = cv2.norm(current_tvec)
					center_aruco_list.append((center_x,center_y))
					depth_data = depth[center_y, center_x]
					distance_from_rgb_list.append(depth_data)

					list_1 = current_rvec.tolist()
					flat_list.append(list_1)
					angle_aruco_list = [item[0] for item in flat_list]
					rotation_mat, _ = cv2.Rodrigues(current_rvec)
					
					width_aruco_list.append(width)
					cv2.aruco.drawDetectedMarkers(image, corners)

					cv2.putText(image, f"ID: {aruco_id[i]}", (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

					cv2.drawFrameAxes(image,cam_mat, dist_mat, current_rvec[0] , current_tvec[0],2,1)	
					cv2.circle(image, (center_x ,center_y), 2, (255, 0, 0), 6)
			cv2.imshow("Aruco Detection", image)
			cv2.waitKey(1) 
		
			return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids, tvec
		except Exception as e:
			print(e)
			
class aruco_tf(Node):

	def __init__(self):

		super().__init__('aruco_tf_publisher')
		self.cv_image = None  
		self.depth_image = None
		self.aruco_markers_msg = ArucoMarkers()

		self.color_cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.colorimagecb, 10)
		self.depth_cam_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depthimagecb, 10)
		self.aruco_markers_pub = self.create_publisher(ArucoMarkers, '/aruco_markers', 10)

		image_processing_rate = 3                                               
		self.bridge = CvBridge()                                                        
		self.tf_buffer = tf2_ros.buffer.Buffer()                                       
		self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
		self.br = tf2_ros.TransformBroadcaster(self)   		                                 
		self.timer = self.create_timer(image_processing_rate, self.process_image)
		self.get_logger().info("Aruco tf publisher node started")   

	def depthimagecb(self, data):

		br = CvBridge()

		self.depth_image = br.imgmsg_to_cv2(data, data.encoding)


	def colorimagecb(self, data):

		br = CvBridge()

		self.cv_image = br.imgmsg_to_cv2(data, "bgr8")
		self.gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

	def process_image(self):
		
		sizeCamX = 1280
		sizeCamY = 720
		centerCamX = 640 
		centerCamY = 360
		focalX = 931.1829833984375
		focalY = 931.1829833984375
		
		try:
			center_list, distance_list, angle_list, width_list, ids,tvec = detect_aruco(self.gray, self.depth_image)
			rpy = []
			list_rpy = []
			self.aruco_markers_msg.marker_ids = []
			self.aruco_markers_msg.poses = []
			for i in range(len(ids)):
				self.aruco_markers_msg.marker_ids.append(ids[i])
				aruco_id = ids[i]
				distance = distance_list[i]
				angle_aruco = angle_list[i]

				angle_aruco = ((0.788*angle_aruco[2]) - ((angle_aruco[2]**2)/3160))
				roll, pitch, yaw = 0, 0, angle_aruco

				rpy.append(roll)
				rpy.append(pitch)
				rpy.append(yaw)
				list_rpy.append(rpy)
				rpy_np = np.array(list_rpy)
				rot_mat, _ = cv2.Rodrigues(rpy_np)

				transform_matrix = np.array([[0, 0, -1],
									[0, 1, 0],
									[-1, 0, 0]])
				
				good_mat = np.dot(rot_mat, transform_matrix)

				euler_good = tf_transformations.euler_from_matrix(good_mat)

				euler_list = euler_good
				roll = euler_list[0]
				pitch = euler_list[1]
				yaw = euler_list[2]
				list_rpy = []
				rpy = []

				q_rot = quaternion_from_euler(roll,pitch,yaw)

				r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
				quaternion = r.as_quat()

				qy = q_rot[1]
				qz = q_rot[2]
				qx = q_rot[0]
				qw = q_rot[3]

				x = tvec[i][0][0]
				y = tvec[i][0][1]
				z = tvec[i][0][2]
				y_1 = (distance * (sizeCamX - center_list[i][0] - centerCamX) / focalX)/1000
				z_1 = (distance * (sizeCamY - center_list[i][1] - centerCamY) / focalY)/1000
				x_1 = distance/1000

				print(x_1, y_1, z_1)

				transform_msg = TransformStamped()
				transform_msg.header.stamp = self.get_clock().now().to_msg()
				transform_msg.header.frame_id = 'camera_link'
				transform_msg.child_frame_id = f'cam_{aruco_id}'
				transform_msg.transform.translation.x = float(x_1)
				transform_msg.transform.translation.y = float(y_1)
				transform_msg.transform.translation.z = float(z_1)
				transform_msg.transform.rotation.x = qx
				transform_msg.transform.rotation.y = qy
				transform_msg.transform.rotation.z = qz
				transform_msg.transform.rotation.w = qw
				self.br.sendTransform(transform_msg)



				try:
					t = self.tf_buffer.lookup_transform('link_base', transform_msg.child_frame_id, rclpy.time.Time())
					transform_msg.header.stamp = self.get_clock().now().to_msg()

					transform_msg.header.frame_id = 'link_base'
					transform_msg.child_frame_id = f'obj_{aruco_id}'
					transform_msg.transform.translation.x = t.transform.translation.x
					transform_msg.transform.translation.y = t.transform.translation.y
					transform_msg.transform.translation.z = t.transform.translation.z
					transform_msg.transform.rotation.x = t.transform.rotation.x
					transform_msg.transform.rotation.y = t.transform.rotation.y
					transform_msg.transform.rotation.z = t.transform.rotation.z
					transform_msg.transform.rotation.w = t.transform.rotation.w
					roll_1 , pitch_1 , yaw_1  = euler_from_quaternion([t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w])
					print(aruco_id)
					self.get_logger().info(f'Id is {aruco_id}')
					self.br.sendTransform(transform_msg)

					pose = Pose()
					pose.position.x = t.transform.translation.x
					pose.position.y = t.transform.translation.y
					pose.position.z = t.transform.translation.z
					pose.orientation.x = t.transform.rotation.x
					pose.orientation.y = t.transform.rotation.y
					pose.orientation.z = t.transform.rotation.z
					pose.orientation.w = t.transform.rotation.w

					self.aruco_markers_msg.poses.append(pose)
					
				except Exception as e:
					pass
			self.aruco_markers_msg.header.stamp = self.get_clock().now().to_msg()
			self.aruco_markers_pub.publish(self.aruco_markers_msg)
		except Exception as e:
			self.get_logger().error(f'Error in processing image: {e}')

def main():
	rclpy.init(args=sys.argv)
	aruco_tf_class = aruco_tf()
	rclpy.spin(aruco_tf_class)
	aruco_tf_class.destroy_node()
	rclpy.shutdown()                                      


if __name__ == '__main__':
	main()