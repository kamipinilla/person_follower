"""
This module provides image processing utilities.
"""
import argparse
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from image_object import ImageObject


def detect_people(image):
    """
    Receives an image and returns the people detected.
    It uses a HOG descriptor with a person detector integrated within OpenCV.

    Args:
        image (ndarray): image to process.

    Returns:
        Detected people in the image as ImageObject's.
    """
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    all_rects, confidences = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    selected_rects = []
    threshold = .9
    for index, (x, y, w, h) in enumerate(all_rects):
        if confidences[index] > threshold:
            image_rect = ImageObject(x, y, w, h)
            selected_rects.append(image_rect)
    selected_rects = apply_non_max_suppression(selected_rects)
    return selected_rects


def apply_non_max_suppression(img_objs):
    """Applies the non-max suppression algorithm to the img_objs, preserving the most appropriate objects."""
    rects_for_imutils = []
    for img_obj in img_objs:
        x, y, w, h = img_obj.as_tuple()
        rects_for_imutils.append([x, y, x + w, y + h])
    rects_for_imutils = np.array(rects_for_imutils)
    picked_rects = non_max_suppression(rects_for_imutils, overlapThresh=0.65)
    picked_img_objs = [ImageObject(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in picked_rects]
    return picked_img_objs


def draw_object(img, img_obj):
    """Draws the img_obj on the img."""
    x, y, w, h = img_obj.as_tuple()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


def main():
    """Main method for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", nargs="?", const="samples/two_people.jpg",
                        help="path of the image to be processed")
    parser.add_argument("--camera", "-c", nargs="?", const="http://admin:admin@157.253.173.229/video.mjpg",
                        help="the URL of the IP camera attached to the robot")
    args = parser.parse_args()

    if args.image and args.camera:
        raise parser.error("Only one input source is allowed")

    if args.image:
        image = cv2.imread(args.image)
        people_img_objs = detect_people(image)
        for img_obj in people_img_objs:
            draw_object(image, img_obj)
        cv2.imshow("Detection", image)
        cv2.waitKey(0)
    else:
        stream_source = args.camera if args.camera is not None else 0
        while cv2.waitKey(1) != 27:
            stream = cv2.VideoCapture(stream_source)
            stream.grab()
            has_frame, image = stream.read()
            if not has_frame:
                print("No image")
                break
            people_img_objs = detect_people(image)
            for img_obj in people_img_objs:
                draw_object(image, img_obj)
            cv2.imshow("Detection", image)


if __name__ == "__main__":
    main()
