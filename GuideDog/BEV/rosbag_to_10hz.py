#!/usr/bin/env python

import rosbag
import rospy
from rospy import Time

input_bag_path = '/home/guide/dataset_guidedog/up/output.bag-_2023-08-22-17-09-25.bag'
output_bag_path = '/home/guide/GuideDog/data/output1.bag'

# Create a new bag for writing
with rosbag.Bag(output_bag_path, 'w') as outbag:
    # Define the desired time interval between messages for 10 Hz
    desired_interval = rospy.Duration(0.1)  # 10 Hz

    # Initialize variables to track last written timestamp for each topic
    last_written_time = {}

    # Initialize variables to store the next messages for each topic
    next_messages = {}

    # Open the input bag
    with rosbag.Bag(input_bag_path, 'r') as bag:
        # Read the input bag
        for topic, msg, t in bag.read_messages():
            if topic in ['/ecparm', '/imu', '/points_raw', '/usb_cam/image_raw/compressed',
                         '/usb_cam0/image_raw0/compressed', '/usb_cam2/image_raw2/compressed',
                         '/usb_cam4/image_raw4/compressed', '/usb_cam6/image_raw6/compressed']:
                # Initialize the last_written_time for the topic if not already set
                if topic not in last_written_time:
                    last_written_time[topic] = t
                    next_messages[topic] = None

                # Calculate the time difference between the current message and the last written message
                time_diff = t - last_written_time[topic]

                # Check if the time difference is greater than or equal to the desired interval
                if time_diff >= desired_interval:
                    if next_messages[topic] is not None:
                        outbag.write(topic, next_messages[topic][1], next_messages[topic][0])
                        last_written_time[topic] = next_messages[topic][0]
                        next_messages[topic] = None
                    else:
                        outbag.write(topic, msg, t)
                        last_written_time[topic] = t

                # Record the closest message to the desired interval
                if next_messages[topic] is None or abs(t - last_written_time[topic] - desired_interval) < abs(next_messages[topic][0] - last_written_time[topic] - desired_interval):
                    next_messages[topic] = (t, msg)

# Print a message when processing is done
print("Processing complete!")
