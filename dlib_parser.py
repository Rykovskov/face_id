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

cnn_model_path = os.path.join(APP_DIR, "mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
#predictor_path = os.path.join(APP_DIR, "shape_predictor_68_face_landmarks.dat")
predictor_path = os.path.join(APP_DIR, "shape_predictor_5_face_landmarks.dat")
model_path = os.path.join(APP_DIR, "dlib_face_recognition_resnet_model_v1.dat")

sp = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

conn = psycopg2.connect(dbname='trafik', user='max', password='tv60hu02', host='localhost')
sql_select_humans = """select id_humans, actions.id_actions, dt_actions, humans.patch_to_pic 
                        from humans 
                        left join actions on actions.id_actions=humans.id_actions 
                        where dlib_parser=false"""

sql_insert_face = """insert into faces (id_actions, id_humans, patch_to_pic, keypoints) values (%s, %s, %s, %s);"""

sql_update_humans_dlib = """update humans set dlib_parser=true where id_humans=%s"""

cur = conn.cursor()
cur.execute(sql_select_humans)
records = cur.fetchall()
if len(records) == 0:
    cur.close()
    conn.close()
    exit()
for (id_humans, id_actions, dt, patch_to_pic) in records:
    if os.path.isfile(patch_to_pic):
        print("Date ------ ", dt)
        frame = cv2.imread(patch_to_pic)
        find_object = 0
        #dets = detector(frame, 1)
        dets = cnn_face_detector(frame, 1)
        s_images = []
        for k, d in enumerate(dets):
            x_face1 = d.rect.left() - 40
            x_face2 = d.rect.right() + 40
            y_face1 = d.rect.top() - 60
            y_face2 = d.rect.bottom() + 60
            if x_face1 < 0:
                x_face1 = 0
            if y_face1 < 0:
                y_face1 = 0
            if x_face2 > 1980:
                x_face2 = 1980
            if y_face2 > 1024:
                y_face2 = 1024
            s_images.append(frame[y_face1:y_face2, x_face1:x_face2])
        num_faces = len(dets)
        if num_faces > 0:
            print("Find face! ", num_faces, dt)
            cur.execute(sql_update_humans_dlib, (id_humans,))
            conn.commit()
            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(sp(frame, detection))
            # Get the aligned face images
            # Optionally:
            images = dlib.get_face_chips(frame, faces, size=640, padding=0.25)
            # images = dlib.get_face_chips(rgb_image, faces, size=640, padding=0.25)
            i = 0
            year_str = dt.strftime("%Y")
            month_str = dt.strftime("%m")
            day_str = dt.strftime("%d")
            for image in images:
                try:
                    # Make directory if not exist
                    fullPath = os.path.join(FACES_DIR, year_str, month_str, day_str)
                    if not os.path.exists(fullPath):
                        os.makedirs(fullPath)
                    filename = dt.strftime("%H_%M_%S") + ".jpg"
                    fullPath = os.path.join(fullPath, filename)
                    dets1 = detector(image, 1)
                    for k1, d1 in enumerate(dets1):
                        shape1 = sp(image, d1)
                    face_descriptor1 = facerec.compute_face_descriptor(image, shape1)
                    out_dump1 = pickle.dumps(face_descriptor1, 1)
                    cur.execute(sql_insert_face, (id_actions, id_humans, fullPath, psycopg2.Binary(out_dump1)))
                    cv2.imwrite(fullPath, s_images[i])
                    conn.commit()
                    print("Save image complete!!")
                except psycopg2.Error as e:
                    print(e.pgerror)
                    print(e.diag.message_detail)
                    print('Ошибка:\n', traceback.format_exc())
                i = i + 1

