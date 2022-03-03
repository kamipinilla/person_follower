#!/usr/bin/env python
"""
This is the main module for the application, it is a ROS node.
It receives the current frame from the camera, communicates with the other modules
and publishes velocity commands to the robot.
"""
import argparse
import sys
import cv2
import imutils
import rospy
from geometry_msgs.msg import Twist

import follower
import multi_pose_recognizer
import yolo

# This is the person that the robot is following, represented as an ImageObject.
tracked_person = None

# This is the Skeleton of the person who has the chosen pose.
# It is used to find the corresponding ImageObject of that person.
recognized_skeleton = None

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", help="the URL of the IP camera attached to the robot",
                    default="http://admin:admin@157.253.173.229/video.mjpg")
myargv = rospy.myargv(argv=sys.argv)[1:]
args = parser.parse_args(myargv)
source_url = args.camera

# Camera stream from which the frames are read.
stream = cv2.VideoCapture(source_url)

# /RosAria/cmd_vel topic, to publish velocity commands.
cmd_vel_topic = rospy.Publisher("/RosAria/cmd_vel", Twist, queue_size=1)

# The name of this ROS node is 'robot_navigator'
rospy.init_node("robot_navigator", anonymous=True)
# rate = rospy.Rate(10000)


def start():
    """
    Starts this ROS node, reading from the camera as frequently as possible.
    Esc key to exit.
    """
    while not rospy.is_shutdown() and cv2.waitKey(1) != 27 and process_next_frame():
        # rate.sleep()
        pass


def process_next_frame():
    """
    Processes each frame from the camera.

    It first looks for the pose of the people in the image, using the pose recognizer module.
    Once a person with the chosen pose is found, their skeleton is used to find their corresponding ImageObject,
    using the person detection module. From this point, the follower module is used to get the velocity command
    to be sent to the robot, according to the position of the tracked person.
    """
    global recognized_skeleton

    if tracked_person is None:
        pose_stream = cv2.VideoCapture(source_url)
        pose_stream.grab()
        has_frame, frame = pose_stream.read()
        frame = resize_input_frame(frame)
        if not has_frame:
            return False

        if recognized_skeleton is None:
            skeleton = multi_pose_recognizer.recognize_pose(frame)
            if skeleton is not None:
                recognized_skeleton = skeleton

        if recognized_skeleton is not None:
            wrap_skeleton(frame)
        # cv2.imshow("Camera", frame)

    if tracked_person is not None:
        # detect_people_stream = cv2.VideoCapture(source_url)
        stream.grab()
        has_frame, frame = stream.read()
        frame = resize_input_frame(frame)
        if not has_frame:
            return False
        img_objs = yolo.detect_people(frame)
        display_objects(frame, img_objs)
        updated_tracked_person = update_tracked_person(img_objs)
        draw_object(frame, tracked_person, color=(0, 255, 255))
        if updated_tracked_person:
            set_robot_velocity(frame)
        else:
            stop_robot()
        cv2.imshow("Camera", frame)

    return True


def set_robot_velocity(frame):
    """
    Obtains the velocity command to be sent to the robot, according to the position of
    the tracked person, relative to the current frame. Then, it publishes the velocity command.
    """
    velocity_command = follower.get_velocity_command(tracked_person, frame)
    publish_velocity_command(velocity_command)


def stop_robot():
    """Stops the robot by publishing a command with 0 linear and angular velocity."""
    publish_velocity_command((0, 0))


def publish_velocity_command(velocity_command):
    """
    Publishes the velocity command to the /RosAria/cmd_vel topic.
    The message is sent using the Twist message type, which encapsulates linear and angular velocity.
    """
    twist_msg = Twist()
    linear_speed, angular_speed = velocity_command
    twist_msg.linear.x = linear_speed
    twist_msg.angular.z = angular_speed
    cmd_vel_topic.publish(twist_msg)


def resize_input_frame(frame):
    """Resizes the frame to enforce a maximum width"""
    return imutils.resize(frame, width=min(400, frame.shape[1]))


def update_tracked_person(img_objs):
    """
    Updates the tracked person if one of the img_obj's in the most recent frame
    proves to be the tracked person, but in a slightly different position.
    """
    global tracked_person
    for img_obj in img_objs:
        if represents_same_tracked_person(img_obj):
            tracked_person = img_obj
            return True
    return False


def represents_same_tracked_person(img_obj):
    """
    Indicates whether the img_obj, obtained from the most recent frame,
    represents the tracked person, but in a slightly different position.
    """
    margin_percent = 1.5

    tracked_person_mid_width = tracked_person.w / 2
    tracked_person_margin = int(tracked_person_mid_width * margin_percent)
    tracked_person_cx = tracked_person.cx
    tracked_person_cy = tracked_person.cy
    tracked_person_min_x = tracked_person_cx - tracked_person_margin
    tracked_person_max_x = tracked_person_cx + tracked_person_margin
    tracked_person_min_y = tracked_person_cy - tracked_person_margin
    tracked_person_max_y = tracked_person_cy + tracked_person_margin

    img_obj_cx = img_obj.cx
    img_obj_cy = img_obj.cy
    is_same_horizontal = tracked_person_min_x < img_obj_cx < tracked_person_max_x
    is_same_vertical = tracked_person_min_y < img_obj_cy < tracked_person_max_y

    tracked_person_area = tracked_person.w * tracked_person.h
    img_obj_area = img_obj.w * img_obj.h
    area_ratio = float(img_obj_area) / tracked_person_area

    threshold_margin = .5
    low_threshold = 1 - threshold_margin
    high_threshold = 1 + threshold_margin
    is_same_ratio = low_threshold < area_ratio < high_threshold

    return is_same_horizontal and is_same_vertical and is_same_ratio


def wrap_skeleton(img):
    """Assigns the tracked person to the ImageObject that corresponds to the recognized Skeleton."""
    global tracked_person
    img_objs = yolo.detect_people(img)
    for img_obj in img_objs:
        if is_skeleton_wrapper(img_obj):
            tracked_person = img_obj
            break


def is_skeleton_wrapper(img_obj):
    """Indicates whether the img_obj corresponds to the recognized Skeleton."""
    hip_x, hip_y = recognized_skeleton.hip_center()
    x, y, w, h = img_obj.as_tuple()
    return x < hip_x < x + w and y < hip_y < y + h


def display_objects(img, img_objs):
    """Displays the img_objs in the img."""
    for img_obj in img_objs:
        draw_object(img, img_obj)


def draw_object(img, img_obj, color=(0, 0, 255)):
    """Draws the img_obj in the img, as a rectangle, with the selected color."""
    x, y, w, h = img_obj.as_tuple()
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=3)


if __name__ == "__main__":
    start()
