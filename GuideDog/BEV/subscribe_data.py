#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2, CompressedImage, Imu
# if need compressed image
#from sensor_msgs.msg import CompressedImage
from message_filters import ApproximateTimeSynchronizer, Subscriber
import callback_pointcloud
import callback_image

def callback(img1_msg, img2_msg, img3_msg, img4_msg, points_msg):
    print("happy")
    # stacked_matrix = callback_pointcloud.callback_pointcloud(points_msg)
    # print(stacked_matrix.shape)
    # callback_image.callback_image(img2_msg)


if __name__ == "__main__":
    rospy.init_node("multi_modal_sync_node")

    image1_sub = Subscriber("/usb_cam0/image_raw0/compressed", CompressedImage) # Mul-cameras
    image2_sub = Subscriber("/usb_cam2/image_raw2/compressed", CompressedImage)
    image3_sub = Subscriber("/usb_cam4/image_raw4/compressed", CompressedImage)
    image4_sub = Subscriber("/usb_cam6/image_raw6/compressed", CompressedImage)
    


    pointcloud_sub = Subscriber("/points_raw", PointCloud2) # Lidar


    

    sync = ApproximateTimeSynchronizer(
        [image1_sub, image2_sub, image3_sub, image4_sub, pointcloud_sub],
        queue_size=10,
        slop=0.15
    )
    sync.registerCallback(callback)


    rospy.spin()

    #控制频率在10HZ
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     rate.sleep()

    rospy.loginfo("Node has been terminated.")