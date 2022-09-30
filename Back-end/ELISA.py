import sys
import cv2
import os
import torch
import torch.backends.cudnn as cudnn
import argparse
import time
from pathlib import Path
from PyQt5 import QtCore
import math
import re
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import pymysql
from datetime import datetime

from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

from_class = uic.loadUiType("pyqt5_server.ui")[0]


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

#server_on = True:On, False:Off
server1_on = False
server2_on = False
server3_on = False
server4_on = False

#Auto_switch 1:On, 0:Off
Auto_switch = 0

class CaptureIpCameraFramesWorker(QThread):
    # Signal emitted when a new image or a new frame is ready.
    ImageUpdated = pyqtSignal(QImage)

    def yolov5_detect(self,
                      weights=ROOT / 'multi_40.pt',  # model.pt path(s)
                      source=ROOT / 'streams.txt',  # file/dir/URL/glob, 0 for webcam
                      data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                      imgsz=(640, 640),  # inference size (height, width)
                      conf_thres=0.005,  # confidence threshold
                      iou_thres=0.45,  # NMS IOU threshold
                      max_det=1,  # maximum detections per image
                      device=' ',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                      view_img=False,  # show results
                      save_txt=False,  # save results to *.txt
                      save_conf=False,  # save confidences in --save-txt labels
                      save_crop=False,  # save cropped prediction boxes
                      nosave=False,  # do not save images/videos
                      classes=None,  # filter by class: --class 0, or --class 0 2 3
                      agnostic_nms=False,  # class-agnostic NMS
                      augment=False,  # augmented inference
                      visualize=False,  # visualize features
                      update=False,  # update all models
                      project=ROOT / 'runs/detect',  # save results to project/name
                      name='exp',  # save results to project/name
                      exist_ok=False,  # existing project/name ok, do not increment
                      line_thickness=1,  # bounding box thickness (pixels)
                      hide_labels=False,  # hide labels
                      hide_conf=False,  # hide confidences
                      half=False,  # use FP16 half-precision inference
                      dnn=False,  # use OpenCV DNN for ONNX inference
                      ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

        start_time = time.time()
        frame_count = 0

        for path, im, im0s, vid_cap, s in dataset:
            frame_count += 1
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                px_tmp = 320
                py_tmp = 200

                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if Auto_switch == 1:#Auto_switch == 1 이면 Motor_control(Auto Mode) 실행
                                self.mt.add_label(label)
                            annotator.box_label(xyxy, label, color=colors(c, True))#Bounding Box Annotate

                            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])) #Bounding Box 좌표 추출
                            rec1 = (int(p1[0] + 13), int(p1[1] + 17))
                            rec2 = (int(p1[0] + 23), int(p1[1] + 32))

                            im0 = annotator.result()
                            cv2.rectangle(im0, rec1, rec2, (255, 255, 255), 1)#Hue Value 측정 영역 표시

                            #Hue Value 측정 좌표
                            px_tmp = int(p1[0] + 13)
                            py_tmp = int(p1[1] + 17)

                            im0 = annotator.result()
                            rgbImage = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                            r = 0
                            g = 0
                            b = 0
                            cv2.rectangle(im0, (px_tmp, py_tmp), (px_tmp + 9, py_tmp + 9), (255, 255, 255), 1)
                            for y in range(px_tmp + 1, px_tmp + 8, 1):  # you need to change the axix value here
                                for x in range(py_tmp + 1, py_tmp + 8, 1):  # you need to change the axix value here
                                    color = rgbImage[x, y]
                                    r += color[0]
                                    g += color[1]
                                    b += color[2]
                                    sum = r + g + b
                                    R = ((r / 64) * (100 / sum))
                                    G = ((g / 64) * (100 / sum))
                                    B = ((b / 64) * (100 / sum))
                                    RG = R - G
                                    RB = R - B
                                    GB = G - B
                                    RG_sq = math.pow(RG, 2)

                                    num = (RG + RB) / 2

                                    decim = math.sqrt(RG_sq + (RB * GB))
                                    H_tmp = (num / decim)
                                    self.average_hue = math.acos(H_tmp)

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                rgbImage = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                r = 0
                g = 0
                b = 0
                cv2.rectangle(im0, (px_tmp, py_tmp), (px_tmp + 9, py_tmp + 9), (255, 255, 255), 1)
                for y in range(px_tmp + 1, px_tmp + 8, 1):  # you need to change the axix value here
                    for x in range(py_tmp + 1, py_tmp + 8, 1):  # you need to change the axix value here
                        color = rgbImage[x, y]
                        r += color[0]
                        g += color[1]
                        b += color[2]
                        sum = r + g + b
                        R = ((r / 64) * (100 / sum))
                        G = ((g / 64) * (100 / sum))
                        B = ((b / 64) * (100 / sum))
                        RG = R - G
                        RB = R - B
                        GB = G - B
                        RG_sq = math.pow(RG, 2)

                        num = (RG + RB) / 2

                        decim = math.sqrt(RG_sq + (RB * GB))
                        H_tmp = (num / decim)
                        self.average_hue = math.acos(H_tmp)


                if view_img:
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

                    h, w, c = im0.shape
                    qImg0 = QtGui.QImage(im0.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                    qt_rgb_image_scaled = qImg0.scaled(1280, 720, Qt.KeepAspectRatio)  # 720p
                    self.ImageUpdated.emit(qt_rgb_image_scaled)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            self.fps = frame_count / (time.time() - start_time)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    def __init__(self, url) -> None: #Camera Thread set
        super(CaptureIpCameraFramesWorker, self).__init__()
        self.url = url
        self.__thread_active = True
        self.fps = 0
        self.__thread_pause = False
        self.mt = MotorControl()

    def run(self) -> None: #YOLOv5 Object detection run
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if cap.isOpened():
            while self.__thread_active:
                if not self.__thread_pause:
                    ret, frame = cap.read()
                    self.yolov5_detect(source=self.url)

        # When everything done, release the video capture object.
        cap.release()
        # Tells the thread's event loop to exit with return code 0 (success).
        self.quit()

    def stop(self) -> None:
        self.__thread_active = False

    def pause(self) -> None:
        self.__thread_pause = True

    def unpause(self) -> None:
        self.__thread_pause = False

class MotorControl: #Motor_Control(Auto_Mode) Class
    def __init__(self):

        self.STATE_COUNT = 5
        self.cnt = [0] * 3  #cnt[0]: filled state 유지 Check 변수, cnt[1]: saved_time 변수, cnt[2]: filled_time sequence 변수(filled_time: 180, 2400, 600)
        print('ELISA sequence START')
        self.cnt[2] = 0
        #Chamber state Check용 변수 선언
        self.EMPTY = "EMPTY"
        self.EMPTY_STATE = [self.EMPTY] * self.STATE_COUNT

        self.FILLED = "FILLED"
        self.FILLED_STATE = [self.FILLED] * self.STATE_COUNT

        self.ENTERING = "ENTERING"
        self.ENTERING_STATE = [self.ENTERING] * self.STATE_COUNT

        self.LEAVING = "LEAVING"
        self.LEAVING_STATE = [self.LEAVING] * self.STATE_COUNT

        self.check_list = []
        self.current_state = "first_filled"


    def add_label(self, label): #Detected class(label)에 따라 motor control
        #stage에 따라 filled_time변경
        if self.cnt[2] == 0:
            self.filled_time = 180
            print(f'first_stage_sequence, filled_time:{self.filled_time}')
        elif self.cnt[2] == 1:
            self.filled_time = 2400
            print(f'second_stage_sequence, filled_time:{self.filled_time}')
        elif self.cnt[2] == 2:
            self.filled_time = 600
            print(f'third_stage_sequence, filled_time::{self.filled_time}')
        else:
            #Auto Mode 종료
            self.filled_time = 0
            print(f'ELISA sequence finished')
            global Auto_switch
            Auto_switch = 0

        word = " ".join(re.findall("[a-zA-Z]+", label)) #label변수에서 class 이름 추출한 후 check_list에 입력
        self.check_list.append(word)
        print(f'check_list:{self.check_list}')

        if self.current_state == "first_filled": #Chameber 상태가 Filled로 인식되기 전, first_sequence 실행
            print(f'Entering_first_sequence')
            self.first_sequence()
        elif self.current_state == "other_filled": #Chameber 상태가 Filled로 인식된 후, second_sequence 실행
            print(f'Entering_second_sequence')
            self.second_sequence()

        if len(self.check_list) > 4:
            self.check_list.pop(0)


    def first_sequence(self):
        # check list에 Empty 또는 Entering문자열이 5번 연속 삽입되었을 때, motor value -0.02(유체를 Chameber로 insert)
        if (self.check_list == self.EMPTY_STATE) | (self.check_list == self.ENTERING_STATE) | (self.check_list == self.LEAVING_STATE):
            if server1_on is True: #Platform1 motor_control
                if servo1.value is not None:
                    print(f'server1_Motor_inhale')
                    servo1.value = servo1.value - 0.02
                    print(f'servo1_value:{servo1.value}')

                    if servo1.value < -1:
                        servo1.value = servo1.min()

            if server2_on is True: #Platform2 motor_control
                if servo2.value is not None:
                    print(f'server2_Motor_inhale')
                    servo2.value = servo2.value - 0.02
                    print(f'servo2_value:{servo2.value}')

                    if servo2.value < -1:
                        servo2.value = servo2.min()

            if server3_on is True: #Platform3 motor_control
                if servo3.value is not None:
                    print(f'server3_Motor_inhale')
                    servo3.value = servo3.value - 0.02
                    print(f'servo3_value:{servo3.value}')

                    if servo3.value < -1:
                        servo3.value = servo3.min()

            if server4_on is True: #Platform4 motor_control
                if servo4.value is not None:
                    print(f'server4_Motor_inhale')
                    servo4.value = servo4.value - 0.02
                    print(f'servo4_value:{servo4.value}')

                    if servo4.value < -1:
                        servo4.value = servo4.min()

        # check_list에 filled문자열이 5번 연속 삽입된 상태가 지속되었을 때 filled_time(sec) 경과후 second_sequence 돌입
        if self.check_list == self.FILLED_STATE:
            self.cnt[0] = self.cnt[0] + 1
            if (self.cnt[0] % 5) == 0:
                self.cnt[1] = time.time()  # Filled가 최소 10frame 이상 확인되었을때, 최초시간 cnt[1] 저장
                print(f'Time saved')
                while time.time() < self.cnt[1] + self.filled_time:  # filled_time만큼 지속후 다음 코드 실행
                    print('Loading. Time delayed:%.1f' % (time.time() - self.cnt[1]))
                    pass
                print(f'Entering_second_sequence')
                self.current_state = "other_filled"


    def second_sequence(self):

        # Empty상태가 5번 이상 지속되었을 때 first_sequence로 돌입 후 Auto_switch 일시정지
        if len(self.check_list) == self.STATE_COUNT:
            if self.check_list == self.EMPTY_STATE:
                print(f'Entering_first_sequence')
                self.current_state = "first_filled"
                global Auto_switch
                Auto_switch = 0
                self.cnt[2] = self.cnt[2] + 1

            # check list에 Filled 또는 Leaving문자열이 5번 연속 삽입되었을 때, motor value -0.02(유체를 Chameber로 insert)
            if (self.check_list == self.FILLED_STATE) | (self.check_list == self.ENTERING_STATE) | (self.check_list == self.LEAVING_STATE):
                if server1_on is True: #Platform1 motor_control
                    if servo1.value is not None:
                        print(f'server1_Motor_inhale')
                        servo1.value = servo1.value - 0.02
                        print(f'servo1:{servo1.value}')

                        if servo1.value < -1:
                            servo1.value = servo1.min()

                if server2_on is True: #Platform2 motor_control
                    if servo2.value is not None:
                        print(f'server2_Motor_inhale')
                        servo2.value = servo2.value - 0.02
                        print(f'servo2:{servo2.value}')

                        if servo2.value < -1:
                            servo2.value = servo2.min()

                if server3_on is True: #Platform3 motor_control
                    if servo3.value is not None:
                        print(f'server3_Motor_inhale')
                        servo3.value = servo3.value - 0.02
                        print(f'servo3:{servo3.value}')

                        if servo3.value < -1:
                            servo3.value = servo3.min()

                if server4_on is True: #Platform4 motor_control
                    if servo4.value is not None:
                        print(f'server4_Motor_inhale')
                        servo4.value = servo4.value - 0.02
                        print(f'servo4:{servo4.value}')

                        if servo4.value < -1:
                            servo4.value = servo4.min()


class MouseEvent(QThread):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.statusbar = self.statusBar()


class MyWindow(QMainWindow, from_class): #GUI window set

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #Button Setting
        self.btn_server_1.clicked.connect(self.setServer1)
        self.btn_server_2.clicked.connect(self.setServer2)
        self.btn_server_3.clicked.connect(self.setServer3)
        self.btn_server_4.clicked.connect(self.setServer4)
        self.btn_inhale_1.clicked.connect(self.motor_1_inhale)
        self.btn_exhale_1.clicked.connect(self.motor_1_exhale)
        self.btn_inhale_2.clicked.connect(self.motor_2_inhale)
        self.btn_exhale_2.clicked.connect(self.motor_2_exhale)
        self.btn_inhale_3.clicked.connect(self.motor_3_inhale)
        self.btn_exhale_3.clicked.connect(self.motor_3_exhale)
        self.btn_inhale_4.clicked.connect(self.motor_4_inhale)
        self.btn_exhale_4.clicked.connect(self.motor_4_exhale)
        self.btn_measure_1.clicked.connect(self.measure1)
        self.btn_measure_2.clicked.connect(self.measure2)
        self.btn_measure_3.clicked.connect(self.measure3)
        self.btn_measure_4.clicked.connect(self.measure4)





        self.btn_save1.clicked.connect(self.save)
        self.btn_save_2.clicked.connect(self.save)
        self.btn_save_3.clicked.connect(self.save)
        self.btn_save_5.clicked.connect(self.save)

        self.btn_server_8.clicked.connect(self.save_1)

        self.btn_stop_1.clicked.connect(self.stop1)
        self.btn_stop_2.clicked.connect(self.stop2)
        self.btn_stop_3.clicked.connect(self.stop3)
        self.btn_stop_4.clicked.connect(self.stop4)

        self.btn_start_1.clicked.connect(self.start1)
        self.btn_start_2.clicked.connect(self.start2)
        self.btn_start_3.clicked.connect(self.start3)
        self.btn_start_4.clicked.connect(self.start4)

        self.conn = pymysql.connect(host='localhost', user='root', password='12345678', db='testdb', charset='utf8')


    def setServer1(self): #1번째 입력창에 URL입력했을 경우 Platform1 server1 set
        url = self.ip_server1.text()
        self.CaptureIpCameraFramesWorker_1 = CaptureIpCameraFramesWorker(url)
        self.CaptureIpCameraFramesWorker_1.ImageUpdated.connect(lambda image: self.ShowCamera1(image))
        self.CaptureIpCameraFramesWorker_1.start()
        global ip1
        ip1 = self.ip_server1.text()[7:20]
        self.setmotor1()
        global server1_on
        server1_on = True

    def setServer2(self):#2번째 입력창에 URL입력했을 경우 Platform2 server2 set
        url = self.ip_server2.text()
        global ip2
        ip2 = self.ip_server2.text()[7:20]
        self.CaptureIpCameraFramesWorker_2 = CaptureIpCameraFramesWorker(url)
        self.CaptureIpCameraFramesWorker_2.ImageUpdated.connect(lambda image: self.ShowCamera2(image))
        self.CaptureIpCameraFramesWorker_2.start()
        self.setmotor2()
        global server2_on
        server2_on = True

    def setServer3(self):#3번째 입력창에 URL입력했을 경우 Platform3 server3 set
        url = self.ip_server3.text()
        global ip3
        ip3 = self.ip_server3.text()[7:20]
        self.CaptureIpCameraFramesWorker_3 = CaptureIpCameraFramesWorker(url)
        self.CaptureIpCameraFramesWorker_3.ImageUpdated.connect(lambda image: self.ShowCamera3(image))
        self.CaptureIpCameraFramesWorker_3.start()
        self.setmotor3()
        global server3_on
        server3_on = True

    def setServer4(self):#4번째 입력창에 URL입력했을 경우 Platform4 server4 set
        url = self.ip_server4.text()
        global ip4
        ip4 = self.ip_server4.text()[7:20]
        self.CaptureIpCameraFramesWorker_4 = CaptureIpCameraFramesWorker(url)
        self.CaptureIpCameraFramesWorker_4.ImageUpdated.connect(lambda image: self.ShowCamera4(image))
        self.CaptureIpCameraFramesWorker_4.start()
        self.setmotor4()
        global server4_on
        server4_on = True

    def setmotor1(self):#Platform1 motor set
        global factory1
        global servo1
        factory1 = PiGPIOFactory(host=str(ip1), port='8889')
        servo1 = Servo(18, pin_factory=factory1)
        servo1.max()


    def setmotor2(self):#Platform2 motor set
        global factory2
        global servo2
        factory2 = PiGPIOFactory(host=str(ip2), port='8890')
        servo2 = Servo(18, pin_factory=factory2)
        servo2.max()
        # pass

    def setmotor3(self):#Platform3 motor set
        global factory3
        global servo3
        factory3 = PiGPIOFactory(host=str(ip3), port='8891')
        servo3 = Servo(18, pin_factory=factory3)
        servo3.max()
        # pass

    def setmotor4(self):#Platform4 motor set
        global factory4
        global servo4
        factory4 = PiGPIOFactory(host=str(ip4), port='8892')
        servo4 = Servo(18, pin_factory=factory4)
        servo4.max()
        # pass

    def motor_1_inhale(self):#Platform1 motor inhale button set
        servo1.value = servo1.value - 0.01
        if servo1.value < -1:
            servo1.min()

    def motor_1_exhale(self):#Platform1 motor exhale button set
        # #from motor_1 import servo1
        servo1.value = servo1.value + 0.01
        if servo1.value > 1:
            servo1.max()

    def motor_2_inhale(self):#Platform2 motor inhale button set
        servo2.value = servo2.value - 0.01
        if servo2.value < -1:
            servo2.min()

    def motor_2_exhale(self):#Platform2 motor exhale button set
        # #from motor_1 import servo1
        servo2.value = servo2.value + 0.01
        if servo2.value > 1:
            servo2.max()

    def motor_3_inhale(self):#Platform3 motor inhale button set
        servo3.value = servo3.value - 0.01
        if servo3.value < -1:
            servo3.min()

    def motor_3_exhale(self):#Platform3 motor exhale button set
        # #from motor_1 import servo1
        servo3.value = servo3.value + 0.01
        if servo3.value > 1:
            servo3.max()

    def motor_4_inhale(self):#Platform4 motor inhale button set
        servo4.value = servo4.value - 0.01
        if servo4.value < -1:
            servo4.min()

    def motor_4_exhale(self):#Platform4 motor exhale button set
        # #from motor_1 import servo1
        servo4.value = servo4.value + 0.01
        if servo4.value > 1:
            servo4.max()

    def stop1(self):#Auto_mode stop button
        global Auto_switch
        Auto_switch = 0
        print('Auto_switch_off')

    def stop2(self):#Auto_mode stop button
        global Auto_switch
        Auto_switch = 0
        print('Auto_switch_off')

    def stop3(self):#Auto_mode stop button
        global Auto_switch
        Auto_switch = 0
        print('Auto_switch_off')

    def stop4(self):#Auto_mode stop button
        global Auto_switch
        Auto_switch = 0
        print('Auto_switch_off')

    def start1(self):#Auto_mode start button
        global Auto_switch
        Auto_switch = 1
        print('Auto_switch_on')

    def start2(self):#Auto_mode start button
        global Auto_switch
        Auto_switch = 1
        print('Auto_switch_on')

    def start3(self):#Auto_mode start button
        global Auto_switch
        Auto_switch = 1
        print('Auto_switch_on')

    def start4(self):#Auto_mode start button
        global Auto_switch
        Auto_switch = 1
        print('Auto_switch_on')

    def save_1(self):
        #id,name,location 입력값 받아오기
        self.id = self.ip_server1_2.text()
        self.name = self.ip_server1_3.text()
        self.location = self.ip_server1_4.text()

        #DataBase연결
        cursor = self.conn.cursor()

        #location_num 가져오는 query문 입력
        sql2 = f'SELECT num FROM local_num WHERE location = \'{self.location}\''

        #location_num 변수 가져오기
        cursor.execute(sql2)
        result = cursor.fetchone()

        sql = "INSERT INTO covid (id, name, location, datetime, concentration, result, local_result)" \
              f"VALUES ({self.id}, '{self.name}', '{self.location}', '{datetime.now()}', 'NULL', {0}, {int(result[0]) + 1})"

        #id, name, location, datetime, location_num변수 DataBase에 입력
        cursor.execute(sql)

        #DataBase 종료
        self.conn.commit()
        self.conn.close()

    def save(self):
        #save self.cen to mysql
        if self.cen > 0.007 :
            self.result = 1
        else:
            self.result = 0

        cursor = self.conn.cursor()

        sql = f"UPDATE covid SET result = {self.result}, concentration = '{self.cen:.1f}'ng/ml WHERE id = {self.id}"
        cursor.execute(sql)

        self.conn.commit()
        self.conn.close()


    def measure1(self):#Hue value 에 따른 Positive, Negative, Concentration 판별
        if (self.CaptureIpCameraFramesWorker_1.average_hue >= 1) :
            mywindow.label_9.setText('POSITIVE')
            mywindow.label_9.setStyleSheet("Color : red")
        else:
            mywindow.label_9.setText('NEGATIVE')
            mywindow.label_9.setStyleSheet("Color : blue")

        self.cen = self.CaptureIpCameraFramesWorker_1.average_hue * 0.028 #(0.007/0.25)
        mywindow.lcd_sat_1.display(self.CaptureIpCameraFramesWorker_1.average_hue * 0.028)

    def measure2(self):#Hue value 에 따른 Positive, Negative 판별
        if (self.CaptureIpCameraFramesWorker_2.average_hue >= 1):
            mywindow.label_10.setText('POSITIVE')
            mywindow.label_10.setStyleSheet("Color : red")
        else:
            mywindow.label_10.setText('NEGATIVE')
            mywindow.label_10.setStyleSheet("Color : blue")

        self.cen = self.CaptureIpCameraFramesWorker_2.average_hue * 0.028  # (0.007/0.25)
        mywindow.lcd_sat_2.display(self.CaptureIpCameraFramesWorker_2.average_hue * 0.028)

    def measure3(self):#Hue value 에 따른 Positive, Negative 판별
        if (self.CaptureIpCameraFramesWorker_3.average_hue >= 1):
            mywindow.label_12.setText('POSITIVE')
            mywindow.label_12.setStyleSheet("Color : red")
        else:
            mywindow.label_12.setText('NEGATIVE')
            mywindow.label_12.setStyleSheet("Color : blue")

        self.cen = self.CaptureIpCameraFramesWorker_3.average_hue * 0.028  # (0.007/0.25)
        mywindow.lcd_sat_3.display(self.CaptureIpCameraFramesWorker_3.average_hue * 0.028)

    def measure4(self):#Hue value 에 따른 Positive, Negative 판별
        if (self.CaptureIpCameraFramesWorker_4.average_hue >= 1):
            mywindow.label_11.setText('POSITIVE')
            mywindow.label_11.setStyleSheet("Color : red")
        else:
            mywindow.label_11.setText('NEGATIVE')
            mywindow.label_11.setStyleSheet("Color : blue")

        self.cen = self.CaptureIpCameraFramesWorker_4.average_hue * 0.028  # (0.007/0.25)
        mywindow.lcd_sat_4.display(self.CaptureIpCameraFramesWorker_4.average_hue * 0.028)

    @QtCore.pyqtSlot()
    def ShowCamera1(self, frame: QImage) -> None: #Platform1 Camera set
        pixmap = QPixmap.fromImage(frame).scaledToWidth(1024).scaledToHeight(768)
        mywindow.label_cam_1.setPixmap(pixmap)
        mywindow.lcd_fps_1.display(self.CaptureIpCameraFramesWorker_1.fps)
        mywindow.lcd_hue_1.display(self.CaptureIpCameraFramesWorker_1.average_hue)

    @QtCore.pyqtSlot()
    def ShowCamera2(self, frame: QImage) -> None:#Platform2 Camera set
        pixmap = QPixmap.fromImage(frame).scaledToWidth(1024).scaledToHeight(768)
        mywindow.label_cam_2.setPixmap(pixmap)
        mywindow.lcd_fps_2.display(self.CaptureIpCameraFramesWorker_2.fps)
        mywindow.lcd_hue_2.display(self.CaptureIpCameraFramesWorker_2.average_hue)

    @QtCore.pyqtSlot()
    def ShowCamera3(self, frame: QImage) -> None:#Platform3 Camera set
        pixmap = QPixmap.fromImage(frame).scaledToWidth(1024).scaledToHeight(768)
        mywindow.label_cam_3.setPixmap(pixmap)
        mywindow.lcd_fps_3.display(self.CaptureIpCameraFramesWorker_3.fps)
        mywindow.lcd_hue_3.display(self.CaptureIpCameraFramesWorker_3.average_hue)

    @QtCore.pyqtSlot()
    def ShowCamera4(self, frame: QImage) -> None:#Platform4 Camera set
        pixmap = QPixmap.fromImage(frame).scaledToWidth(1024).scaledToHeight(768)
        mywindow.label_cam_4.setPixmap(pixmap)
        mywindow.lcd_fps_4.display(self.CaptureIpCameraFramesWorker_4.fps)
        mywindow.lcd_hue_4.display(self.CaptureIpCameraFramesWorker_4.average_hue)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()

    app.exec_()
