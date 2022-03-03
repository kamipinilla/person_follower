#!/usr/bin/env python
"""
This module allows controlling the robot with the keyboard.

This is not part of the person follower solution. Rather, it can be used as an isolate ROS node
that permits controlling the movement of the robot with the keyboard.
The name of the node is robot_controller. It receives input from the keyboard and
publishes Twist messages to the /RosAria/cmd_vel topic.

Use WASD keys to move the robot, as well as QEZX keys for diagonal movements.
"""
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist


def robot_controller():
    """Starts the ROS node, listens to keyboard events and publishes velocity commands."""
    pub = rospy.Publisher("/RosAria/cmd_vel", Twist, queue_size=1)
    rospy.init_node("robot_controller", anonymous=True)
    rate = rospy.Rate(1000)
    while not rospy.is_shutdown():
        # A blank image is used to receive keyboard events with OpenCV.
        blank_img = np.zeros([100, 100, 3], dtype=np.uint8)
        blank_img.fill(255)
        cv2.imshow("Read", blank_img)
        key = cv2.waitKey(1)
        twist = Twist()
        straight_vel = .3
        angular_vel = np.pi / 12
        if key == ord("w"):
            rospy.loginfo("UP")
            twist.linear.x = straight_vel
        elif key == ord("s"):
            rospy.loginfo("DOWN")
            twist.linear.x = -straight_vel
        elif key == ord("a"):
            rospy.loginfo("LEFT")
            twist.angular.z = angular_vel
        elif key == ord("d"):
            rospy.loginfo("RIGHT")
            twist.angular.z = -angular_vel
        elif key == ord("q"):
            rospy.loginfo("UP_LEFT")
            twist.linear.x = straight_vel
            twist.angular.z = angular_vel
        elif key == ord("e"):
            rospy.loginfo("UP_RIGHT")
            twist.linear.x = straight_vel
            twist.angular.z = -angular_vel
        elif key == ord("z"):
            rospy.loginfo("DOWN_LEFT")
            twist.linear.x = -straight_vel
            twist.angular.z = angular_vel
        elif key == ord("x"):
            twist.linear.x = -straight_vel
            twist.angular.z = -angular_vel
        elif key == 27:  # Esc key to exit
            break
        else:
            twist.linear.x = 0
            twist.angular.z = 0
        pub.publish(twist)
        rate.sleep()


if __name__ == "__main__":
    try:
        robot_controller()
    except rospy.ROSInterruptException:
        pass
