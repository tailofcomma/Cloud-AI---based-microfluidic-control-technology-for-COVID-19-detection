import cv2
import glob
from time import sleep
import uuid
import os
import numpy as np

BASE_PATH = 'C:/Users/LK/Desktop/test_entering' # 불러올 폴더
IMAGE_PATH = BASE_PATH + '/*.jpg' # 폴더안의 모든 사진 불러오기
CROP_PATH = 'C:/Users/LK/Desktop/circle_crop/' # 저장폴더위치


def parse_imageName(path):
    return os.path.basename(path)


def save_image(path, frame):
    imageName = parse_imageName(path)
    writePath = CROP_PATH + imageName
    cv2.imwrite(writePath, frame)
    print(writePath)


def crop_image(path, param2):
    if param2 > 100:
        return

    frame = cv2.imread(path)
    cimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 60, param1=40, param2=param2, minRadius=0, maxRadius=0)


    if circles is None:
        crop_image(path, param2 + 1)

    for circle in circles:
        circle = np.uint16(np.around(circles))

        y1 = int(circle[0][0][1]) - circle[0][0][2]
        y2 = int(circle[0][0][1]) + circle[0][0][2]
        x1 = int(circle[0][0][0]) - circle[0][0][2]
        x2 = int(circle[0][0][0]) + circle[0][0][2]

        if x1 < 0:
            x1 = 0
        if x2 > cimg.shape[1]:
            x2 = cimg.shape[1]
        if y1 < 0:
            y1 = 0
        if y2 > cimg.shape[0]:
            y2 = cimg.shape[0]

        crop_frame = frame[y1:y2, x1:x2]

        if crop_frame.shape[0] + crop_frame.shape[1] <= 500:
            save_image(path, crop_frame)
            return


def main():
    image_paths = glob.glob(IMAGE_PATH)
    if image_paths is not None:
        for path in image_paths:
            crop_image(path, 1)
    else:
        print("images_list is None")


main()