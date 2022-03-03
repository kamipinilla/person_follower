"""
This module handles person detection. It is based on the YOLOv3 algorithm from the DarkNet project.
"""
import argparse
import os
import cv2
import numpy as np
import rospkg

from image_object import ImageObject

yolov3_config_file = "yolo/yolov3.cfg"
yolov3_weights_file = "yolo/yolov3.weights"

yolov3_tiny_config_file = "yolo/yolov3-tiny.cfg"
yolov3_tiny_weights_file = "yolo/yolov3-tiny.weights"

yolov2_voc_config_file = "yolo/yolov2-voc.cfg"
yolov2_voc_weights_file = "yolo/yolov2-voc.weights"

yolov2_tiny_voc_config_file = "yolo/yolov2-tiny-voc.cfg"
yolov2_tiny_voc_weights_file = "yolo/yolov2-tiny-voc.weights"

# Classes file with the name of the classes that the model can detect.
# It is only of interest the 'person' label from this file.
classes_file = "yolo/yolov3.txt"

if not __name__ == "__main__":
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('person_follower')

    # Configuration and weights files for the yolov3 model.
    yolov3_config_file = os.path.join(package_path, "src/scripts", yolov3_config_file)
    yolov3_weights_file = os.path.join(package_path, "src/scripts", yolov3_weights_file)

    # Configuration and weights files for the yolov3-tiny model.
    yolov3_tiny_config_file = os.path.join(package_path, "src/scripts", yolov3_tiny_config_file)
    yolov3_tiny_weights_file = os.path.join(package_path, "src/scripts", yolov3_tiny_weights_file)

    # Configuration and weights files for the yolov2-voc model.
    yolov2_voc_config_file = os.path.join(package_path, "src/scripts", yolov2_voc_config_file)
    yolov2_voc_weights_file = os.path.join(package_path, "src/scripts", yolov2_voc_weights_file)

    # Configuration and weights files for the yolov2-tiny-voc model.
    yolov2_tiny_voc_config_file = os.path.join(package_path, "src/scripts", yolov2_tiny_voc_config_file)
    yolov2_tiny_voc_weights_file = os.path.join(package_path, "src/scripts", yolov2_tiny_voc_weights_file)

    classes_file = os.path.join(package_path, "src/scripts", classes_file)

# Selected configuration and weights
config_file = yolov3_tiny_config_file
weights_file = yolov3_tiny_weights_file

person_label = "person"
with open(classes_file, 'r') as f:
    # List of the available classes for detection. Only the 'person' class is of interest.
    classes = [line.strip() for line in f.readlines()]

# Neural network for person detection, using the selected configuration and weights files.
nn = cv2.dnn.readNet(weights_file, config_file)


def detect_people(image):
    """Receives an image and returns the people detected, as ImageObject's."""
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392

    # start_time = time.time()
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    nn.setInput(blob)

    output_layers = nn.forward(get_output_layers(nn))
    # print("{:.3f}".format(time.time() - start_time))
    confidences = []
    boxes = []
    conf_threshold = .05
    nms_threshold = .2

    for out in output_layers:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if classes[int(class_id)] == person_label:
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    confidences.append(float(confidence))
                    boxes.append((x, y, w, h))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    people_img_objs = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        img_obj = ImageObject(x, y, w, h)
        people_img_objs.append(img_obj)
    return people_img_objs


def get_output_layers(network):
    """Returns the output layers of the neural network, where the detections are found."""
    layer_names = network.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    return output_layers


def draw_object(img, img_obj):
    """Draws the img_obj on the img as a rectangle."""
    x, y, w, h = img_obj.as_tuple()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


def main():
    """Main method for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", nargs="?", const="samples/single.jpg",
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
        stream = cv2.VideoCapture(stream_source)
        while cv2.waitKey(1) != 27:
            stream.grab()
            has_frame, image = stream.read()
            if not has_frame:
                break
            people_img_objs = detect_people(image)
            for img_obj in people_img_objs:
                draw_object(image, img_obj)
            cv2.imshow("Detection", image)


if __name__ == '__main__':
    main()
