#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
#import dlib
import numpy as np
import time
import sys
import traceback
import datetime
import mrcnn.config
import mrcnn.utils
from pathlib import Path
from mrcnn.model import MaskRCNN
import psycopg2
from psycopg2.extensions import register_adapter, AsIs


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)

APP_DIR = Path("/home/max/face_id")
ROOT_DIR = Path("/home/max/base")
CAR_DIR = os.path.join(ROOT_DIR, "cars")
HUMAN_DIR = os.path.join(ROOT_DIR, "humans")
PET_DIR = os.path.join(ROOT_DIR, "pets")
delta_size_x = 40
delta_size_y = 90
size_pic_with = 1920
size_pic_high = 1080
find_object = 0

# Directory to save logs and trained model
MODEL_DIR = os.path.join(APP_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(APP_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


def get_car_boxes(boxes, class_ids):
    car_boxes = []
    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [2, 3, 8, 6, 4,]:
            car_boxes.append(box)

    return np.array(car_boxes)


def get_human_boxes(boxes, class_ids):
    human_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [1, ]:
            human_boxes.append(box)

    return np.array(human_boxes)


def get_pet_boxes(boxes, class_ids):
    pet_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [16, 17, 18]:
            pet_boxes.append(box)

    return np.array(pet_boxes)

def normal_rect(y1, x1, y2, x2):
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

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

conn = psycopg2.connect(dbname='trafik', user='max', password='tv60hu02', host='localhost')
sql_select_actions = """select id_actions, dt_actions, patch_to_pic from actions where MaskRCNN=false order by dt_actions;"""

sql_update_actions_MaskRCNN = """update actions set MaskRCNN=true where id_actions=%s"""
sql_update_actions_car = """update actions set Detect_car=true where id_actions=%s"""
sql_update_actions_human = """update actions set Detect_human=true where id_actions=%s"""
sql_update_actions_pet = """update actions set Detect_pet=true where id_actions=%s"""

sql_insert_car = """insert into cars (id_actions, patch_to_pic) values (%s, %s);"""
sql_insert_human = """insert into humans (id_actions, patch_to_pic) values (%s, %s);"""
sql_insert_pet = """insert into pets (id_actions, patch_to_pic) values (%s, %s);"""

cur = conn.cursor()
cur.execute(sql_select_actions)
records = cur.fetchall()
if len(records) == 0:
    cur.close()
    conn.close()
    exit()

for (id_record, dt, path_to_file) in records:
    if os.path.isfile(path_to_file):
        #File exist
        frame = cv2.imread(path_to_file)
        find_object = 0
        # Run the image through the Mask R-CNN model to get results.
        results = model.detect([frame], verbose=0)
        t2 = time.time()
        # print("Time detect all object:", (t2 - t1))
        r = results[0]
        car_boxes = get_car_boxes(r['rois'], r['class_ids'])
        if len(car_boxes) > 0:
            find_object = 1
        human_boxes = get_human_boxes(r['rois'], r['class_ids'])
        if len(human_boxes) > 0:
            find_object = 1
        pet_boxes = get_pet_boxes(r['rois'], r['class_ids'])
        if len(pet_boxes) > 0:
            find_object = 1
        if find_object:
            cur.execute(sql_update_actions_MaskRCNN, (id_record,))
            conn.commit()
        # Draw each box on the frame
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%m")
        day_str = dt.strftime("%d")
        i = 0
        for box in car_boxes:
            i = i + 1
            print("Car: ", box)
            y1, x1, y2, x2 = box
            #y1, x1, y2, x2 = normal_rect(y1, x1, y2, x2)
            # Make directory if not exist
            fullPath = os.path.join(CAR_DIR, year_str, month_str, day_str)
            if not os.path.exists(fullPath):
                os.makedirs(fullPath)
            filename = dt.strftime("%H_%M_%S_%f_") + str(i) + ".jpg"
            # Save image to disk
            img_car = frame[y1:y2, x1:x2]
            fullPath = os.path.join(fullPath, filename)
            cv2.imwrite(fullPath, img_car)
            cur.execute(sql_update_actions_car, (id_record,))
            cur.execute(sql_insert_car, (id_record, fullPath))
            conn.commit()
        i = 0
        for box in human_boxes:
            i = i + 1
            print("Human: ", box)
            y1, x1, y2, x2 = box
            #y1, x1, y2, x2 = normal_rect(y1, x1, y2, x2)
            # Make directory if not exist
            fullPath = os.path.join(HUMAN_DIR, year_str, month_str, day_str)
            if not os.path.exists(fullPath):
                os.makedirs(fullPath)
            filename = dt.strftime("%H_%M_%S_%f_") + str(i) + ".jpg"
            # Save image to disk
            img_human = frame[y1:y2, x1:x2]
            fullPath = os.path.join(fullPath, filename)
            cv2.imwrite(fullPath, img_human)
            cur.execute(sql_update_actions_human, (id_record,))
            cur.execute(sql_insert_human, (id_record, fullPath))
            conn.commit()
        i = 0
        for box in pet_boxes:
            i = i + 1
            print("Pet: ", box)
            y1, x1, y2, x2 = box
            #y1, x1, y2, x2 = normal_rect(y1, x1, y2, x2)
            # Make directory if not exist
            fullPath = os.path.join(PET_DIR, year_str, month_str, day_str)
            if not os.path.exists(fullPath):
                os.makedirs(fullPath)
            filename = dt.strftime("%H_%M_%S_%f_") + str(i) + ".jpg"
            # Save image to disk
            img_pet = frame[y1:y2, x1:x2]
            fullPath = os.path.join(fullPath, filename)
            cv2.imwrite(fullPath, img_pet)
            cur.execute(sql_update_actions_pet, (id_record,))
            cur.execute(sql_insert_pet, (id_record, fullPath))
            conn.commit()

cur.close()
conn.close()