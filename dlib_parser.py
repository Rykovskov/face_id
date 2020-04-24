#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import cv2
import dlib
import datetime
import sys
import traceback
import numpy as np
import time
import _pickle as pickle
from scipy.spatial import distance
import psycopg2
from psycopg2.extensions import register_adapter, AsIs


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, descriptors[i])
        ++i
        temp_array.append(temp)
    return temp_array


def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                    _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)


ROOT_DIR = Path("/home/max/base")
APP_DIR = Path("/home/max/face_id")
FACES_DIR = os.path.join(ROOT_DIR, "faces")

predictor_path = os.path.join(APP_DIR, "shape_predictor_68_face_landmarks.dat")
model_path = os.path.join(APP_DIR, "dlib_face_recognition_resnet_model_v1.dat")
facerec = dlib.face_recognition_model_v1(model_path)
sp = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

conn = psycopg2.connect(dbname='trafik', user='max', password='tv60hu02', host='localhost')
sql_select_humans = """select id_humans, id_actions, patch_to_pic from humans where dlib_parser=false;"""

cur = conn.cursor()
cur.execute(sql_select_humans)
records = cur.fetchall()
if len(records) == 0:
    cur.close()
    conn.close()
    exit()
for (id_humans, id_actions, patch_to_pic) in records:
    if os.path.isfile(patch_to_pic):
        frame = cv2.imread(patch_to_pic)
        find_object = 0
