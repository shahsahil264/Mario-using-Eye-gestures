import cv2
import dlib
from math import hypot
import numpy as np
import os
from pynput.keyboard import Key, Controller
import time

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/sahil/Desktop/HCI/shape_predictor_68_face_landmarks.dat")
#Downloaded the above file from "https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"
keyboard = Controller()
BLINK_THRESH = 4.2
THRESHOLD_THRESH = 30

def get_landmarks(keypoint_img):
    landmarks = predictor(gray, face)
    for i in range(68):
        cv2.circle(keypoint_img, (landmarks.part(i).x,
                                  landmarks.part(i).y), 2, 255, -1)
    return landmarks


def get_left_eye_landmarks(landmarks):
    left_eye_left_pt = (landmarks.part(36).x, landmarks.part(36).y)
    left_eye_right_pt = (landmarks.part(39).x, landmarks.part(39).y)
    left_eye_top_mid = (int((landmarks.part(37).x + landmarks.part(38).x) / 2),
                        int((landmarks.part(37).y + landmarks.part(38).y) / 2))
    left_eye_bottom_mid = (int((landmarks.part(
        40).x + landmarks.part(41).x) / 2), int((landmarks.part(40).y + landmarks.part(41).y) / 2))
    return left_eye_left_pt, left_eye_right_pt, left_eye_top_mid, left_eye_bottom_mid


def get_right_eye_landmarks(landmarks):
    right_eye_left_pt = (landmarks.part(42).x, landmarks.part(42).y)
    right_eye_right_pt = (landmarks.part(45).x, landmarks.part(45).y)
    right_eye_top_mid = (int((landmarks.part(43).x + landmarks.part(44).x) / 2),
                         int((landmarks.part(43).y + landmarks.part(44).y) / 2))
    right_eye_bottom_mid = (int((landmarks.part(47).x + landmarks.part(46).x) / 2),
                            int((landmarks.part(47).y + landmarks.part(46).y) / 2))
    return right_eye_left_pt, right_eye_right_pt, right_eye_top_mid, right_eye_bottom_mid


def get_eye_position(eye_left_pt, eye_right_pt, eye_top_mid, eye_bottom_mid):
    hor_line = cv2.line(
        keypoint_img, eye_left_pt, eye_right_pt, (0, 255, 0), 1)
    ver_line = cv2.line(
        keypoint_img, eye_top_mid, eye_bottom_mid, (0, 255, 0), 1)

    hor_line_len = hypot(
        (eye_left_pt[0] - eye_right_pt[0]), (eye_left_pt[1] - eye_right_pt[1]))
    ver_line_len = hypot(
        (eye_top_mid[0] - eye_bottom_mid[0]), (eye_top_mid[1] - eye_bottom_mid[1]))
    return hor_line, ver_line, hor_line_len, ver_line_len


def get_eye_thresh(eye_region):
    min_x, min_y, max_x, max_y = np.min(eye_region[:, 0]), np.min(eye_region[:, 1]), np.max(eye_region[:, 0]), np.max(
        eye_region[:, 1])

    eye = cv2.resize(
        flip[min_y:max_y, min_x:max_x], None, fx=5, fy=5)
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, thresh_eye = cv2.threshold(
        gray_eye, THRESHOLD_THRESH, 255, cv2.THRESH_BINARY)
    h, w = thresh_eye.shape
    thresh_eye_half = thresh_eye[0:h, 0:int(w / 2)]
    return gray_eye, thresh_eye, thresh_eye_half


# Decide direction where mario will move
def decide_direction(r_nz, l_nz, r, l, avg_open_eye_ratio):
    os.system("clear")
    ctrl = ""
    if r_nz > l_nz:
        ctrl = "Right"
        keyboard.press(Key.right)

        print("CONTROL : " + ctrl)
    else:
        ctrl = "Left"
        keyboard.press(Key.left)

        print("CONTROL : " + ctrl)
    cv2.imshow("EYE INFO RIGHT", r)
    cv2.imshow("EYE INFO LEFT", l)

    print("AVG OPEN EYE RATIO : " + str(round(avg_open_eye_ratio, 3)))
    print("Right non zero : " + str(l_nz))
    print("left non zero : " + str(r_nz))


while 1:
    ret, frame = cap.read()
    flip = cv2.flip(frame, 1)
    keypoint_img = flip.copy()
    gray = cv2.cvtColor(flip, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = get_landmarks(keypoint_img)
        # LEFT EYEBALL
        left_eye_left_pt, left_eye_right_pt, left_eye_top_mid, left_eye_bottom_mid = get_left_eye_landmarks(landmarks)
        left_hor_line, left_ver_line, left_hor_line_len, left_ver_line_len = get_eye_position(left_eye_left_pt,
                                                                                              left_eye_right_pt,
                                                                                              left_eye_top_mid,
                                                                                              left_eye_bottom_mid)

        left_open_eye_ratio = left_hor_line_len / left_ver_line_len
        temp_left = []
        for i in range(36, 42):
            temp_left.append((landmarks.part(i).x, landmarks.part(i).y))
        left_eye_region = np.array(temp_left, np.int32)
        left_gray_eye, left_thresh_eye, left_thresh_eye_half = get_eye_thresh(left_eye_region)

        # RIGHT EYEBALL
        right_eye_left_pt, right_eye_right_pt, right_eye_top_mid, right_eye_bottom_mid = get_right_eye_landmarks(
            landmarks)

        right_hor_line, right_ver_line, right_hor_line_len, right_ver_line_len = get_eye_position(right_eye_left_pt,
                                                                                                  right_eye_right_pt,
                                                                                                  right_eye_top_mid,
                                                                                                  right_eye_bottom_mid)

        right_open_eye_ratio = right_hor_line_len / right_ver_line_len
        temp_right = []
        for i in range(42, 48):
            temp_right.append((landmarks.part(i).x, landmarks.part(i).y))
        right_eye_region = np.array(temp_right, np.int32)
        right_gray_eye, right_thresh_eye, right_thresh_eye_half = get_eye_thresh(right_eye_region)

        # INFO ABOUT THE EYE

        r = np.hstack((right_gray_eye, right_thresh_eye))
        l = np.hstack((left_gray_eye, left_thresh_eye))

        l_nz = cv2.countNonZero(left_thresh_eye_half)
        r_nz = cv2.countNonZero(right_thresh_eye_half)

        avg_open_eye_ratio = (right_open_eye_ratio + left_open_eye_ratio) / 2

        # DISPLAY
        decide_direction(r_nz, l_nz, r, l, avg_open_eye_ratio)

        # Check if eye is blinking
        if avg_open_eye_ratio > BLINK_THRESH:
            keyboard.press(Key.space)
            time.sleep(0.5)
            keyboard.release(Key.space)
            print("STATUS : BLINKING")
        else:
            print("STATUS : NOT BLINKING")

    cv2.imshow("Facial Keypoints", keypoint_img)
    cv2.imshow("Orignal Image", flip)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
