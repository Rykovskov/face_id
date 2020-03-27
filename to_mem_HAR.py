#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import time
import sys
import traceback
import datetime
import compare_pic
from pathlib import Path

size_pic_with = 1920
size_pic_high = 1080
delta_size_x = 40
delta_size_y = 90
max_broken_frame = 1000
min_dist_pic = 0.6

ROOT_DIR = Path(".")
CAR_DIR = os.path.join(ROOT_DIR, "car_img")
HUMAN_DIR = os.path.join(ROOT_DIR, "human_img")
PET_DIR = os.path.join(ROOT_DIR, "pet_img")

cascade_src = 'cars.xml'

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def normal_rect(x1, y1, x2, y2):
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

try:
    #vcap = cv2.VideoCapture("rtsp://admin@192.168.21.168:554/user=admin&password=&channel=1&stream=0")
    vcap = cv2.VideoCapture("rtsp://admin@192.168.20.93:554/11")
except:
    print("I can not open source video!!!")
    sys.exit()

counter_broken_frame = 0
img_car_old = np.full((300, 400, 3), 130, dtype=np.uint8)
img_human_old = np.full((300, 400, 3), 130, dtype=np.uint8)
img_pet_old = np.full((300, 400, 3), 130, dtype=np.uint8)

while True:
    ret, frame = vcap.read()
    if frame is None:
        counter_broken_frame = counter_broken_frame + 1
        print(counter_broken_frame)
        vcap.release()
        vcap = cv2.VideoCapture("rtsp://admin@192.168.20.93:554/11")
        continue
    if counter_broken_frame > max_broken_frame:
        print("Unable reading source video !!!")
        sys.exit()
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #resc_frame = rescale_frame(frame, percent=80)
    t1 = time.time()