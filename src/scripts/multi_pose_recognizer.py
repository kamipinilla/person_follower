"""
This module handles pose estimation and pose recognition.
"""
import argparse
import os
import cv2
import numpy as np
import rospkg
from enum import IntEnum


class Keypoint(IntEnum):
    """
    Constants representing the keypoints for a person of the Microsoft COCO set.
    """
    NOSE = 0
    NECK = 1
    R_SHOULDER = 2
    R_ELBOW = 3
    R_WRIST = 4
    L_SHOULDER = 5
    L_ELBOW = 6
    L_WRIST = 7
    R_HIP = 8
    R_KNEE = 9
    R_ANKLE = 10
    L_HIP = 11
    L_KNEE = 12
    L_ANKLE = 13
    R_EYE = 14
    L_EYE = 15
    R_EAR = 16
    L_EAR = 17


# Number of keypoints for a person.
n_points = len(Keypoint.__members__.items())


class Skeleton(object):
    """
    Holds the coordinates for the keypoints of a person.
    A coordinate is None if it is not present in the person upon estimation.
    """
    def __init__(self):
        self.nose = None
        self.neck = None
        self.r_shoulder = None
        self.r_elbow = None
        self.r_wrist = None
        self.l_shoulder = None
        self.l_elbow = None
        self.l_wrist = None
        self.r_hip = None
        self.r_knee = None
        self.r_ankle = None
        self.l_hip = None
        self.l_knee = None
        self.l_ankle = None
        self.r_eye = None
        self.l_eye = None
        self.r_ear = None
        self.l_ear = None

    def to_keypoints(self):
        """Transforms the Skeleton representation to a list of keypoints indexed by the Keypoint enum constants."""
        keypoints = [None] * n_points
        keypoints[Keypoint.NOSE] = self.nose
        keypoints[Keypoint.NECK] = self.neck
        keypoints[Keypoint.R_SHOULDER] = self.r_shoulder
        keypoints[Keypoint.R_ELBOW] = self.r_elbow
        keypoints[Keypoint.R_WRIST] = self.r_wrist
        keypoints[Keypoint.L_SHOULDER] = self.l_shoulder
        keypoints[Keypoint.L_ELBOW] = self.l_elbow
        keypoints[Keypoint.L_WRIST] = self.l_wrist
        keypoints[Keypoint.R_HIP] = self.r_hip
        keypoints[Keypoint.R_KNEE] = self.r_knee
        keypoints[Keypoint.R_ANKLE] = self.r_ankle
        keypoints[Keypoint.L_HIP] = self.l_hip
        keypoints[Keypoint.L_KNEE] = self.l_knee
        keypoints[Keypoint.L_ANKLE] = self.l_ankle
        keypoints[Keypoint.R_EYE] = self.r_eye
        keypoints[Keypoint.L_EYE] = self.l_eye
        keypoints[Keypoint.R_EAR] = self.r_ear
        keypoints[Keypoint.L_EAR] = self.l_ear
        return keypoints

    @classmethod
    def from_keypoints(cls, keypoints):
        """Builds a Skeleton representation for the list of keypoints."""
        skeleton = cls()
        skeleton.nose = keypoints[Keypoint.NOSE]
        skeleton.neck = keypoints[Keypoint.NECK]
        skeleton.r_shoulder = keypoints[Keypoint.R_SHOULDER]
        skeleton.r_elbow = keypoints[Keypoint.R_ELBOW]
        skeleton.r_wrist = keypoints[Keypoint.R_WRIST]
        skeleton.l_shoulder = keypoints[Keypoint.L_SHOULDER]
        skeleton.l_elbow = keypoints[Keypoint.L_ELBOW]
        skeleton.l_wrist = keypoints[Keypoint.L_WRIST]
        skeleton.r_hip = keypoints[Keypoint.R_HIP]
        skeleton.r_knee = keypoints[Keypoint.R_KNEE]
        skeleton.r_ankle = keypoints[Keypoint.R_ANKLE]
        skeleton.l_hip = keypoints[Keypoint.L_HIP]
        skeleton.l_knee = keypoints[Keypoint.L_KNEE]
        skeleton.l_ankle = keypoints[Keypoint.L_ANKLE]
        skeleton.r_eye = keypoints[Keypoint.R_EYE]
        skeleton.l_eye = keypoints[Keypoint.L_EYE]
        skeleton.r_ear = keypoints[Keypoint.R_EAR]
        skeleton.l_ear = keypoints[Keypoint.L_EAR]
        return skeleton

    def hip_center(self):
        """Returns the coordinates of the hip center, if existent in the Skeleton."""
        if self.l_hip is None or self.r_hip is None:
            return None
        l_hip_x, l_hip_y = self.l_hip
        r_hip_x, r_hip_y = self.r_hip
        x = (l_hip_x + r_hip_x) / 2.
        y = (l_hip_y + r_hip_y) / 2.
        return x, y


class Pose(object):
    """
    A pose is specified by a series of rules that the Skeleton must have for it to have the pose.

    The Skeleton must satisfy all rules for it to have the pose. A rule is a function that
    receives a Skeleton and indicates whether the Skeleton satisfies that rule.
    """
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def has_pose(self, skeleton):
        for rule in self.rules:
            if not rule(skeleton):
                return False
        return True


def get_pose_1():
    """This is the default pose selected for the project."""
    def has_90_r_armpit(skeleton):
        angle = get_angle(skeleton.r_shoulder, skeleton.r_hip, skeleton.r_elbow)
        return 60 < angle < 120 if angle is not None else False

    def has_90_r_elbow(skeleton):
        angle = get_angle(skeleton.r_elbow, skeleton.r_shoulder, skeleton.r_wrist)
        return 60 < angle < 120 if angle is not None else False

    pose = Pose()
    pose.add_rule(has_90_r_armpit)
    pose.add_rule(has_90_r_elbow)
    return pose


def get_pose_2():
    """This is an example of an alternative pose."""
    def has_r_arm_straight(skeleton):
        angle = get_angle(skeleton.r_elbow, skeleton.r_shoulder, skeleton.r_wrist)
        return angle > 140 if angle is not None else False

    def has_r_arm_lifted(skeleton):
        angle = get_angle(skeleton.r_shoulder, skeleton.r_wrist, skeleton.r_hip)
        return 30 < angle < 65 if angle is not None else False

    pose = Pose()
    pose.add_rule(has_r_arm_straight)
    pose.add_rule(has_r_arm_lifted)
    return pose


# Selected pose, to be compared with estimated poses of the people in the image.
chosen_pose = get_pose_1()

# Path for the neural network architecture.
proto_file = "models/pose/coco/pose_deploy_linevec.prototxt"
# Path for the neural network initial weights.
weights_file = "models/pose/coco/pose_iter_440000.caffemodel"

if not __name__ == "__main__":
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('person_follower')
    proto_file = os.path.join(package_path, "src/scripts", proto_file)
    weights_file = os.path.join(package_path, "src/scripts", weights_file)

# List of valid keypoint pairs.
pose_pairs = [[Keypoint.NECK, Keypoint.R_SHOULDER],
              [Keypoint.NECK, Keypoint.L_SHOULDER],
              [Keypoint.R_SHOULDER, Keypoint.R_ELBOW],
              [Keypoint.R_ELBOW, Keypoint.R_WRIST],
              [Keypoint.L_SHOULDER, Keypoint.L_ELBOW],
              [Keypoint.L_ELBOW, Keypoint.L_WRIST],
              [Keypoint.NECK, Keypoint.R_HIP],
              [Keypoint.R_HIP, Keypoint.R_KNEE],
              [Keypoint.R_KNEE, Keypoint.R_ANKLE],
              [Keypoint.NECK, Keypoint.L_HIP],
              [Keypoint.L_HIP, Keypoint.L_KNEE],
              [Keypoint.L_KNEE, Keypoint.L_ANKLE],
              [Keypoint.NECK, Keypoint.NOSE],
              [Keypoint.NOSE, Keypoint.R_EYE],
              [Keypoint.R_EYE, Keypoint.R_EAR],
              [Keypoint.NOSE, Keypoint.L_EYE],
              [Keypoint.L_EYE, Keypoint.L_EAR],
              [Keypoint.R_SHOULDER, Keypoint.L_EAR],
              [Keypoint.L_SHOULDER, Keypoint.R_EAR]]

# Index of Part Affinity Fields corresponding to the valid keypoint pairs.
map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
           [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
           [47, 48], [49, 50], [53, 54], [51, 52], [55, 56],
           [37, 38], [45, 46]]

# Colors for displaying the keypoints.
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255],
          [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255],
          [0, 0, 255], [255, 0, 0], [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 255]]

# Neural network for the pose estimation.
nn = cv2.dnn.readNetFromCaffe(proto_file, weights_file)


def recognize_pose(image):
    """Receives an image and returns the Skeleton that has the chosen pose, if there is any."""
    frame_width = image.shape[1]
    frame_height = image.shape[0]

    # t = time.time()
    input_blob = create_input_blob(image)
    nn.setInput(input_blob)
    output = nn.forward()
    # print("{:.3f}".format(time.time() - t))

    detected_keypoints, keypoints_list = detect_keypoints(output, frame_width, frame_height)
    valid_pairs, invalid_pairs = get_valid_pairs(output, detected_keypoints, frame_width, frame_height)
    personwise_keypoints_with_score = get_personwise_keypoints_with_score(valid_pairs, invalid_pairs, keypoints_list)
    personwise_keypoints = get_personwise_keypoints(personwise_keypoints_with_score, keypoints_list)

    skeletons = get_skeletons(personwise_keypoints)

    display_results = True
    if display_results:
        display_keypoints(image, skeletons)
        # display_pairs(image, personwise_keypoints_with_score, keypoints_list)
    return skeleton_with_pose(skeletons)


def get_keypoints(prob_map, threshold=0.1):
    """Returns all the keypoints found, with the selected threshold."""
    map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)
    map_mask = np.uint8(map_smooth > threshold)

    contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    keypoints = []
    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_map = map_smooth * blob_mask
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
        keypoints.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

    return keypoints


def get_valid_pairs(output, detected_keypoints, orig_width, orig_height):
    """Returns the valid pairs in the detected keypoints."""
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7

    # loop for every POSE_PAIR
    for k in range(len(map_idx)):
        # A->B constitute a limb
        paf_a = output[0, map_idx[k][0], :, :]
        paf_b = output[0, map_idx[k][1], :, :]
        paf_a = cv2.resize(paf_a, (orig_width, orig_height))
        paf_b = cv2.resize(paf_b, (orig_width, orig_height))

        # Find the keypoints for the first and second limb
        cand_a = detected_keypoints[pose_pairs[k][0]]
        cand_b = detected_keypoints[pose_pairs[k][1]]
        n_a = len(cand_a)
        n_b = len(cand_b)

        if n_a != 0 and n_b != 0:
            valid_pair = np.zeros((0, 3))
            for i in range(n_a):
                max_j = -1
                max_score = -1
                found = 0
                for j in range(n_b):
                    d_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=n_interp_samples),
                                            np.linspace(cand_a[i][1], cand_b[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for l in range(len(interp_coord)):
                        paf_interp.append([paf_a[int(round(interp_coord[l][1])), int(round(interp_coord[l][0]))],
                                           paf_b[int(round(interp_coord[l][1])), int(round(interp_coord[l][0]))]])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[cand_a[i][3], cand_b[max_j][3], max_score]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs


def get_personwise_keypoints_with_score(valid_pairs, invalid_pairs, keypoints_list):
    """
    Returns the vaild keypoints for each person in the image, using the valid pairs,
    the invalid pairs and the list of all keypoints in the image.
    """
    personwise_keypoints = -1 * np.ones((0, n_points+1))

    for k in range(len(map_idx)):
        if k not in invalid_pairs:
            part_as = valid_pairs[k][:, 0]
            part_bs = valid_pairs[k][:, 1]
            index_a, index_b = np.array(pose_pairs[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwise_keypoints)):
                    if personwise_keypoints[j][index_a] == part_as[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwise_keypoints[person_idx][index_b] = part_bs[i]
                    personwise_keypoints[person_idx][-1] += \
                        keypoints_list[part_bs[i].astype(int), 2] + valid_pairs[k][i][2]

                elif not found and k < n_points-1:
                    row = -1 * np.ones(n_points+1)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwise_keypoints = np.vstack([personwise_keypoints, row])
    return personwise_keypoints


def create_input_blob(image):
    """Creates an input blob from the image, to be fed to the neural network as its input."""
    input_size = 300
    return cv2.dnn.blobFromImage(image, 1.0 / 255, (input_size, input_size), (0, 0, 0), swapRB=False, crop=False)


def detect_keypoints(output, orig_width, orig_height):
    """Returns the detected keypoints in the image, based on the neural network's output."""
    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = .1

    for part in range(n_points):
        prob_map = output[0, part, :, :]
        prob_map = cv2.resize(prob_map, (orig_width, orig_height))
        keypoints = get_keypoints(prob_map, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1
        detected_keypoints.append(keypoints_with_id)
    return detected_keypoints, keypoints_list


def display_keypoints(image, skeletons):
    """Display the skeletons' keypoints in the image."""
    frame_clone = image.copy()
    for skeleton in skeletons:
        person_keypoints = skeleton.to_keypoints()
        for index, keypoint in enumerate(person_keypoints):
            if keypoint is not None:
                x, y = keypoint
                cv2.circle(frame_clone, (x, y), 5, colors[index], -1, cv2.LINE_AA)
        if chosen_pose.has_pose(skeleton):
            x, y = skeleton.neck
            cv2.circle(frame_clone, (x, y), 20, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.imshow("Keypoints", frame_clone)


def display_pairs(image, personwise_keypoints, keypoints_list):
    """Displays the valid pairs for each person in the image."""
    frame_clone = image.copy()
    for n in range(len(personwise_keypoints)):
        for i in range(n_points - 1):
            index = personwise_keypoints[n][np.array(pose_pairs[i])]
            if -1 not in index:
                point_a = np.int32(keypoints_list[index.astype(int), 0])
                point_b = np.int32(keypoints_list[index.astype(int), 1])

                cv2.line(frame_clone, (point_a[0], point_b[0]), (point_a[1], point_b[1]), colors[i], 3, cv2.LINE_AA)

    cv2.imshow("Detected Pose", frame_clone)


def get_angle(tail, p1, p2):
    """Returns the shortest angle (in degrees) between the vectors p1 - tail and p2 - tail."""
    if tail is None or p1 is None or p2 is None:
        return None
    tail = np.array(tail)
    p1 = np.array(p1)
    p2 = np.array(p2)

    v1 = p1 - tail
    v2 = p2 - tail

    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    norm_product = norm_v1 * norm_v2
    if norm_product == 0:
        return None
    cos = dot / norm_product
    cos = np.clip(cos, -1, 1)
    angle_rad = np.arccos(cos)
    angle_deg = (angle_rad / np.pi) * 180
    return angle_deg if not np.isnan(angle_deg) else None


def get_person_keypoints(person_keypoints_with_score, keypoints_list):
    """
    Returns the keypoints without the scores.
    The scores need to be removed to create a Skeleton from the keypoints.
    """
    keypoints = []
    person_keypoints_with_score = person_keypoints_with_score[:-1]
    for keypoint_id in person_keypoints_with_score:
        keypoint = [int(coord) for coord in keypoints_list[int(keypoint_id)][0:2].tolist()]
        keypoints.append(keypoint)
    return keypoints


def get_skeletons(personwise_keypoints):
    """Transforms the personwise keypoints into Skeleton representations."""
    num_people = len(personwise_keypoints)
    skeletons = []
    for i in range(num_people):
        person_keypoints = personwise_keypoints[i]
        skeleton = Skeleton.from_keypoints(person_keypoints)
        skeletons.append(skeleton)
    return skeletons


def skeleton_with_pose(skeletons):
    """Returns the skeleton that has the chosen pose, if there is any."""
    for skeleton in skeletons:
        if chosen_pose.has_pose(skeleton):
            return skeleton
    return None


def get_personwise_keypoints(personwise_keypoints_with_score, keypoints_list):
    """
    Returns the personwise keypoints for the people in the image.
    The scores are removed to create a Skeleton from the keypoints.
    """
    personwise_keypoints = []
    for person_keypoints_with_score in personwise_keypoints_with_score:
        person_keypoints = get_person_keypoints(person_keypoints_with_score, keypoints_list)
        personwise_keypoints.append(person_keypoints)
    return personwise_keypoints


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
        img = cv2.imread(args.image)
        recognize_pose(img)
        cv2.waitKey(0)
    else:
        stream_source = args.camera if args.camera is not None else 0
        while cv2.waitKey(1) != 27:
            stream = cv2.VideoCapture(stream_source)
            stream.grab()
            has_frame, frame = stream.read()
            if not has_frame:
                print("No image")
                break
            recognize_pose(frame)


if __name__ == "__main__":
    main()
