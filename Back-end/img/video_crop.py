import cv2
import uuid
import glob
import os
import sys

i = 0
video_list = glob.glob('C:/Users/LK/Desktop/2022_kwix/2022_data_video/taewon-20220320T194508Z-001/taewon/empty_video/empty_1_taewon.mp4')

for video in video_list:
    cap = cv2.VideoCapture(video)
    t = 0
    while (1):
        ret, frame = cap.read()
        t = t + 1
        i = i + 1
        if ret == True:
            if t % 2 == 0:
                cv2.imwrite('C:/Users/LK/Desktop/2022_kwix/2022_data_video/taewon-20220320T194508Z-001/taewon/' + str(i) + "_taewon_" + str(uuid.uuid1()) + '.jpg', frame)
        else:
            os.remove(video)
            break
    cap.release()
print("end")
