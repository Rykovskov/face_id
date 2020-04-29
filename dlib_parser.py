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

ROOT_DIR = Path("/home/max/base")
APP_DIR = Path("/home/max/face_id")
FACES_DIR = os.path.join(ROOT_DIR, "faces")

cnn_model_path = os.path.join(APP_DIR, "mmod_human_face_detector.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
predictor_path = os.path.join(APP_DIR, "shape_predictor_68_face_landmarks.dat")
#predictor_path = os.path.join(APP_DIR, "shape_predictor_5_face_landmarks.dat")
model_path = os.path.join(APP_DIR, "dlib_face_recognition_resnet_model_v1.dat")

sp = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()
facerec = dlib.face_recognition_model_v1(model_path)

conn = psycopg2.connect(dbname='trafik', user='max', password='tv60hu02', host='localhost')
sql_select_humans = """select id_humans, actions.id_actions, dt_actions, humans.patch_to_pic 
                        from humans 
                        left join actions on actions.id_actions=humans.id_actions 
                        where dlib_parser=false"""

sql_insert_face = """insert into faces (id_actions, id_humans, patch_to_pic) values (%s, %s, %s);"""
sql_insert_face_des = """insert into faces (id_actions, id_humans, patch_to_pic, keypoints) values (%s, %s, %s, %s);"""

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
        frame = cv2.imread(patch_to_pic)
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%m")
        day_str = dt.strftime("%d")
        fullPath = os.path.join(FACES_DIR, year_str, month_str, day_str)
        find_object = 0
        #lendmark detector
        descriptors = []
        s_images=[]
        dets = detector(frame, 1)
        num_faces = len(dets)
        if num_faces > 0:
            if not os.path.exists(fullPath):
                os.makedirs(fullPath)
            for k, d in enumerate(dets):
                x_face1 = d.left() - 40
                x_face2 = d.right() + 40
                y_face1 = d.top() - 40
                y_face2 = d.bottom() + 40
                if x_face1 < 0:
                    x_face1 = 0
                if y_face1 < 0:
                    y_face1 = 0
                if x_face2 > 1920:
                    x_face2 = 1920
                if y_face2 > 1080:
                    y_face2 = 1080
                    # Get the landmarks/parts for the face in box d.
                shape = sp(frame, d)
                # Compute the 128D vector that describes the face in img identified by
                # shape.
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                descriptors.append(face_descriptor)
                s_images.append(frame[y_face1:y_face2, x_face1:x_face2])
            i = 0
            for img in s_images:
                filename = dt.strftime("%H_%M_%S_%f_") + str(i) + ".jpg"
                fullPath = os.path.join(fullPath, filename)
                cur.execute(sql_update_humans_dlib, (id_humans,))
                conn.commit()
                out_dump = pickle.dumps(descriptors[i], 1)
                cur.execute(sql_insert_face_des, (id_actions, id_humans, fullPath, (psycopg2.Binary(out_dump))))
                cv2.imwrite(fullPath, img)
                conn.commit()
                i = i + 1
            continue
        #CNN Detector 34
        dets_cnn = cnn_face_detector(frame, 1)
        s_images = []
        for k, d in enumerate(dets_cnn):
            x_face1 = d.rect.left() - 40
            x_face2 = d.rect.right() + 40
            y_face1 = d.rect.top() - 40
            y_face2 = d.rect.bottom() + 40
            if x_face1 < 0:
                x_face1 = 0
            if y_face1 < 0:
                y_face1 = 0
            if x_face2 > 1920:
                x_face2 = 1920
            if y_face2 > 1080:
                y_face2 = 1080
            s_images.append(frame[y_face1:y_face2, x_face1:x_face2])
        num_faces = len(dets_cnn)
        if num_faces > 0:
            year_str = dt.strftime("%Y")
            month_str = dt.strftime("%m")
            day_str = dt.strftime("%d")
            #print("Find face! ", num_faces, dt)
            i = 0
            for img in s_images:
                i = i + 1
                fullPath = os.path.join(FACES_DIR, year_str, month_str, day_str)
                if not os.path.exists(fullPath):
                    os.makedirs(fullPath)
                filename = dt.strftime("%H_%M_%S_%f_") + str(i) + ".jpg"
                fullPath = os.path.join(fullPath, filename)
                cur.execute(sql_update_humans_dlib, (id_humans,))
                conn.commit()
                cur.execute(sql_insert_face, (id_actions, id_humans, fullPath))
                cv2.imwrite(fullPath, img)
                conn.commit()

