# RealsenseData.msg
# Custom message to encapsulate RealSense image data

std_msgs/Header header
sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
float32 depth_scale

