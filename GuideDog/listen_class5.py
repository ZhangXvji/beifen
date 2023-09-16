#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32

def callback(data):
    rospy.loginfo("Received data: %f", data.data)
    # data就是接收到的方向，Float32，0停止，1前进，2后退，3右转，4左转
    # 你可以在callback里处理

def receive_node():
    rospy.init_node('receive_node', anonymous=True)
    rospy.Subscriber('/output_class', Float32, callback)
    rospy.spin()

if __name__ == '__main__':
    receive_node()
