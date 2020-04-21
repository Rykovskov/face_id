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
import imutils
import psycopg2

size_pic_with = 1920
size_pic_high = 1080
delta_size_x = 40
delta_size_y = 90
max_broken_frame = 1000
min_dist_pic = 0.6

ROOT_DIR = Path("/home/max/base")

cascade_car = 'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_car)
cascade_car1 = 'cas1.xml'
car_cascade1 = cv2.CascadeClassifier(cascade_car1)

cascade_car3 = 'cas3.xml'
car_cascade3 = cv2.CascadeClassifier(cascade_car3)
cascade_car4 = 'cas4.xml'
car_cascade4 = cv2.CascadeClassifier(cascade_car4)

cascade_human = 'haarcascade_fullbody.xml'
human_cascade = cv2.CascadeClassifier(cascade_human)
cascade_human1 = 'haarcascade_frontalface_alt_tree.xml'
human_cascade1 = cv2.CascadeClassifier(cascade_human1)
cascade_human2 = 'haarcascade_frontalface_alt.xml'
human_cascade2 = cv2.CascadeClassifier(cascade_human2)


cascade_dog = 'haarcascade_frontalface_alt.xml'
dog_cascade = cv2.CascadeClassifier(cascade_dog)

conn = psycopg2.connect(dbname='trafik', user='max', password='tv60hu02', host='localhost')
sql_insert_actions = """insert into actions (dt_actions, patch_to_pic, name_pic) values (%s, %s, %s)  RETURNING id_actions;"""
sql_insert_rects = """insert into rectangles (id_actions, x1, x2, y1, y2) values (%s, %s, %s, %s, %s);"""

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


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

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

try:
    vcap = cv2.VideoCapture("rtsp://admin@192.168.20.168:554/user=admin&password=&channel=1&stream=0")
    #vcap = cv2.VideoCapture("rtsp://admin@192.168.20.93:554/11")
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
        vcap = cv2.VideoCapture("rtsp://admin@192.168.20.168:554/user=admin&password=&channel=1&stream=0")
        continue
    if counter_broken_frame > max_broken_frame:
        print("Unable reading source video !!!")
        sys.exit()
    t1 = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    fm = variance_of_laplacian(gray)
    if fm > 19000.00:
        # Skip unfocused images
        continue
    font = cv2.FONT_HERSHEY_DUPLEX
    #Car
    rects = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(330, 330), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        rects = car_cascade1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(330, 330),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            rects = car_cascade3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(330, 330),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
            if len(rects) == 0:
                rects = car_cascade4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(330, 330),
                                                          flags=cv2.CASCADE_SCALE_IMAGE)
                if len(rects) == 0:
                    rects = []
                else:
                    rects[:, 2:] += rects[:, :2]
            else:
                rects[:, 2:] += rects[:, :2]
        else:
            rects[:, 2:] += rects[:, :2]
    else:
        rects[:, 2:] += rects[:, :2]
    vis = frame.copy()
    now = datetime.datetime.now()
    for x1, y1, x2, y2 in rects:
        # Make directory if not exist
        year_str = datetime.datetime.now().strftime("%Y")
        month_str = datetime.datetime.now().strftime("%m")
        day_str = datetime.datetime.now().strftime("%d")
        fullPath = os.path.join(ROOT_DIR, year_str, month_str, day_str)
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        filename = datetime.datetime.now().strftime("%H_%M_%S") + ".jpg"
        fullPath = os.path.join(fullPath, filename)
        print("Car ! ", x1, y1, x2, y2)
        (x1, y1, x2, y2) = normal_rect(x1, y1, x2, y2)
        #img_car = frame[y1:y2, x1:x2]
        #roi = gray[y1:y2, x1:x2]
        #vis_roi = vis[y1:y2, x1:x2]
        #draw_rects(vis, rects, (0, 255, 0))
        #cv2.imwrite(filename, img_car)
        cv2.putText(vis, str(fm), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imwrite(fullPath, vis)
        cur = conn.cursor()
        cur.execute(sql_insert_actions, (now, fullPath, filename, ))
        id_of_new_row = cur.fetchone()[0]
        cur.execute(sql_insert_rects, (id_of_new_row, x1, x2, y1, y2,))
        conn.commit()
        cur.close()
    #Human
    rects = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(130, 230),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        rects = human_cascade1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(130, 230),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            rects = human_cascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(130, 230),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            if len(rects) == 0:
                rects = []
            else:
                rects[:, 2:] += rects[:, :2]
        else:
            rects[:, 2:] += rects[:, :2]
    else:
        rects[:, 2:] += rects[:, :2]
    vis = frame.copy()
    for x1, y1, x2, y2 in rects:
        # Make directory if not exist
        year_str = datetime.datetime.now().strftime("%Y")
        month_str = datetime.datetime.now().strftime("%m")
        day_str = datetime.datetime.now().strftime("%d")
        fullPath = os.path.join(ROOT_DIR, year_str, month_str, day_str)
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        filename = datetime.datetime.now().strftime("%H_%M_%S") + ".jpg"
        fullPath = os.path.join(fullPath, filename)
        print("Human ! ", x1, y1, x2, y2)
        (x1, y1, x2, y2) = normal_rect(x1, y1, x2, y2)
        cv2.putText(vis, str(fm), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imwrite(fullPath, vis)
        cur = conn.cursor()
        cur.execute(sql_insert_actions, (now, fullPath, filename,))
        id_of_new_row = cur.fetchone()[0]
        cur.execute(sql_insert_rects, (id_of_new_row, x1, x2, y1, y2,))
        conn.commit()
        cur.close()
    #Pets
    rects = cascade_dog.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(130, 230),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        rects = []
    else:
        rects[:, 2:] += rects[:, :2]
    vis = frame.copy()
    for x1, y1, x2, y2 in rects:
        # Make directory if not exist
        year_str = datetime.datetime.now().strftime("%Y")
        month_str = datetime.datetime.now().strftime("%m")
        day_str = datetime.datetime.now().strftime("%d")
        fullPath = os.path.join(ROOT_DIR, year_str, month_str, day_str)
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        filename = datetime.datetime.now().strftime("%H_%M_%S") + ".jpg"
        fullPath = os.path.join(fullPath, filename)
        print("Human ! ", x1, y1, x2, y2)
        (x1, y1, x2, y2) = normal_rect(x1, y1, x2, y2)
        cv2.putText(vis, str(fm), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imwrite(fullPath, vis)
        cur = conn.cursor()
        cur.execute(sql_insert_actions, (now, fullPath, filename,))
        id_of_new_row = cur.fetchone()[0]
        cur.execute(sql_insert_rects, (id_of_new_row, x1, x2, y1, y2,))
        conn.commit()
        cur.close()


