"""
This module determines the movement command that should be sent
to the robot, according to the position of the tracked object.
"""
import numpy as np


class Direction(object):
    """
    Holds the possible directions and speeds for the linear
    and angular velocity that make up the movement command.
    """
    LINEAR_SPEED = .5
    MINI_LINEAR_SPEED = .05
    ANGULAR_SPEED = np.pi / 12
    MINI_ANGULAR_SPEED = np.pi / 24

    FORWARD = LINEAR_SPEED
    BACK = -LINEAR_SPEED
    LEFT = ANGULAR_SPEED
    RIGHT = -ANGULAR_SPEED
    MINI_FORWARD = MINI_LINEAR_SPEED
    MINI_BACK = -MINI_LINEAR_SPEED
    MINI_LEFT = MINI_ANGULAR_SPEED
    MINI_RIGHT = -MINI_ANGULAR_SPEED


def get_velocity_command(img_obj, img):
    """
    Calculates the movement command according to the position of the tracked object.

    The command consists of both linear and angular speeds.

    Args:
        img_obj (ImageObject): tracked object.
        img (ndarray): full image from the robot's camera.

    Returns:
        command (float, float): it consists of linear and angular components.
    """
    vertical_ratio = get_vertical_ratio(img_obj, img)
    linear_speed = get_linear_speed(vertical_ratio)

    lateral_percentage = get_lateral_percentage(img_obj, img)
    angular_speed = get_angular_speed(lateral_percentage)

    return linear_speed, angular_speed


def get_lateral_percentage(img_obj, img):
    """
    Returns the ratio between the horizontal position of the img_obj's center and the width of img.

    Args:
        img_obj (ImageObject): tracked object.
        img (ndarray): full image from the robot's camera.

    Returns:
        lateral percentage (float)
    """
    cx = img_obj.cx
    img_width = img.shape[1]
    cx_percentage = float(cx) / img_width
    return cx_percentage


def get_vertical_ratio(img_obj, img):
    """
    Returns the ratio between the height of img_obj and img.

    Args:
        img_obj (ImageObject): tracked object.
        img (ndarray): full image from the robot's camera.

    Returns:
        vertical ratio (float)
    """
    img_height = img.shape[0]
    img_obj_height = img_obj.h
    ratio = float(img_obj_height) / img_height
    return ratio


def get_linear_speed(vertical_ratio):
    """Calculates the linear speed according to the vertical ratio of the tracked object."""
    low = .7
    high = .9

    low_mid = low
    mid_high = high
    if vertical_ratio < low:
        return Direction.FORWARD
    elif low <= vertical_ratio < low_mid:
        return Direction.MINI_FORWARD
    elif low_mid <= vertical_ratio < mid_high:
        return 0.
    elif mid_high <= vertical_ratio < high:
        return Direction.MINI_BACK
    elif vertical_ratio >= high:
        return Direction.BACK


def get_angular_speed(lateral_percentage):
    """Calculates the angular speed according to the lateral percentage of the tracked object"""
    wide_margin = .15
    small_margin = .05
    mid = .5

    a = mid - wide_margin
    b = mid - small_margin
    c = mid + small_margin
    d = mid + wide_margin

    if lateral_percentage < a:
        return Direction.LEFT
    elif a <= lateral_percentage < b:
        return Direction.MINI_LEFT
    elif b <= lateral_percentage < c:
        return 0.
    elif c <= lateral_percentage < d:
        return Direction.MINI_RIGHT
    elif lateral_percentage >= d:
        return Direction.RIGHT
