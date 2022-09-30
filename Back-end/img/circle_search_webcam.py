import cv2
import numpy as np

cap = cv2.VideoCapture(0) # 카메라 모듈 사용

while(1):
    ret, frame = cap.read() # 카메라 모듈 연속프레임 읽기

    cv2.imshow("Red", R)
    cv2.imshow("Green", G)
    cv2.imshow("Blue", B)
    cv2.waitKey(0)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # BGR을 HSV로 변환해줌
    # # 파란색 범위
    # lower_blue = np.array([100,100,120])
    # upper_blue = np.array([150,255,255])
    #
    # # 초록색 범위
    # lower_green = np.array([50, 150, 50])
    # upper_green = np.array([80, 255, 255])
    #
    # # 빨간색 범위
    # lower_red = np.array([150, 50, 50])
    # upper_red = np.array([180, 255, 255])
    #
    # # Threshold the HSV image to get only blue colors
    # mask = cv2.inRange(hsv, lower_blue, upper_blue) # 110<->150 Hue(색상) 영역을 지정
    # mask1 = cv2.inRange(hsv, lower_green, upper_green) # 영역 이하는 모두 날림 검정. 그 이상은 모두 흰색 두개로 Mask를 씌움
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame, frame, mask=mask) # 흰색 영역에 파랑색 마스크를 씌워줌
    # res1 = cv2.bitwise_and(frame, frame, mask=mask1) # 흰색 영역에 초록색 마스크를 씌워줌
    # res2 = cv2.bitwise_and(frame, frame, mask=mask2) # 흰색 영역에 빨강색 마스크를 씌워줌
    # chamber_rect = cv2
    #
    # cv2.imshow('frame',frame) # 원본 영상을 보여줌
    # # cv2.imshow('Blue', res) # 마스크 위에 파랑색을 씌운 것을 보여줌
    # # cv2.imshow('Green', res1) # 마스크 위에 초록색을 씌운 것을 보여줌
    # # cv2.imshow('red', res2) # 마스크 위에 빨강색을 씌운 것을 보여줌

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# ESC로 종료
cv2.destroyAllWindows()
