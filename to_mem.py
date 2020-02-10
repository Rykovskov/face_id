#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
import time
import sys
import traceback
import datetime

size_pic_with = 1920
size_pic_high = 1080
delta_size_x = 40
delta_size_y = 90
max_broken_frame = 100

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
def normal_rect(x1,y1,x2,y2):
    x1 = x1 - delta_size_x
    x2 = x2 + delta_size_x
    y1 = y1 - delta_size_y
    y2 = y2 + delta_size_y
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > size_pic_with:
        x2 = size_pic_with
    if y2 > size_pic_high:
        y2 = size_pic_high
    return x1, y1, x2, y2

cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

try:
    vcap = cv2.VideoCapture("rtsp://admin@192.168.21.168:554/user=admin&password=&channel=1&stream=0")
except:
    print("I can not open source video!!!")
    sys.exit()
counter_broken_frame = 0
while True:
    try:
        try:
            ret, frame = vcap.read()
            if frame is None:
                counter_broken_frame = counter_broken_frame + 1
                print(counter_broken_frame)
        except:
            counter_broken_frame = counter_broken_frame + 1
            print(counter_broken_frame)
        if counter_broken_frame > max_broken_frame:
            print("Unable reading source video !!!")
            sys.exit()
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resc_frame = rescale_frame(frame, percent=80)
        t1 = time.time()
        dets = cnn_face_detector(resc_frame, 1)
        if len(dets) > 0:
           for i, d in enumerate(dets):
               print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                       i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
               x_left, y_top, x_right, y_bottom = normal_rect(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
               face_img =resc_frame[y_top:y_bottom, x_left:x_right]
               filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S")+".jpg"
               cv2.imwrite(filename, face_img)
               # filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + "_resc.jpg"
               # cv2.imwrite(filename, resc_frame)
               # filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + "_rgb.jpg"
               # cv2.imwrite(filename, rgb_frame)
               filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + "_org.jpg"
               cv2.imwrite(filename, frame)
           t2 = time.time()
           print("Time:", (t2-t1))
           #rects = dlib.rectangles()
           #rects.extend([d.rect for d in dets])
           #win.clear_overlay()
           #win.set_image(resc_frame)
           #win.add_overlay(rects)
           t3=time.time()
           print("Full Time: ", (t3-t1))
    except:
        print('Ошибка:\n', traceback.format_exc())
        sys.exit()