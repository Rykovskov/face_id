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
import compare_pic

size_pic_with = 1920
size_pic_high = 1080
delta_size_x = 40
delta_size_y = 90
max_broken_frame = 1000

ROOT_DIR = Path(".")
CAR_DIR = os.path.join(ROOT_DIR, "car_img")
HUMAN_DIR = os.path.join(ROOT_DIR, "human_img")
PET_DIR = os.path.join(ROOT_DIR, "pet_img")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
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

#Load Dlib
#cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

try:
    #vcap = cv2.VideoCapture("rtsp://admin@192.168.21.168:554/user=admin&password=&channel=1&stream=0")
    vcap = cv2.VideoCapture("rtsp://admin@192.168.20.93:554/11")
except:
    print("I can not open source video!!!")
    sys.exit()
counter_broken_frame = 0
img_car_old = np.random.rand(430, 200)
img_human_old = np.random.rand(430, 200)
img_pet_old = np.random.rand(430, 200)
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
    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([frame], verbose=0)
    t2 = time.time()
    #print("Time detect all object:", (t2 - t1))
    r = results[0]
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])
    human_boxes = get_human_boxes(r['rois'], r['class_ids'])
    pet_boxes = get_pet_boxes(r['rois'], r['class_ids'])
    # Draw each box on the frame
    for box in car_boxes:
        print("Car: ", box)
        y1, x1, y2, x2 = box
        #Save image to disk
        img_car = frame[y1:y2, x1:x2]
        if img_car_old is not None:
            filename = os.path.join(CAR_DIR, datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + ".jpg")
            t_car1 = time.time()
            # 1
            dist = compare_pic.Get_Difference(img_car, img_car_old)
            t_car2 = time.time()
            print("Pohoge ", dist)
            if dist < 0.8:
                print("Ne Pohoge ", dist)
                cv2.imwrite(filename, img_car)
            print("Time compare pic:", (t_car2 - t_car1))
        img_car_old = img_car
    for box in human_boxes:
        print("Human: ", box)
        y1, x1, y2, x2 = box
        # Save image to disk
        img_human = frame[y1:y2, x1:x2]
        if img_human_old is not None:
            filename = os.path.join(HUMAN_DIR, datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + ".jpg")
            t_car1 = time.time()
            # 1
            dist = compare_pic.Get_Difference(img_human, img_human_old)
            t_car2 = time.time()
            if dist < 0.8:
                print("Ne Pohoge ", dist)
                cv2.imwrite(filename, img_human)
        img_human_old = img_human
    for box in pet_boxes:
        print("Pet: ", box)
        y1, x1, y2, x2 = box
        # Save image to disk
        img_pet = frame[y1:y2, x1:x2]
        if img_pet_old is not None:
            filename = os.path.join(PET_DIR, datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + ".jpg")
            t_car1 = time.time()
            # 1
            dist = compare_pic.Get_Difference(img_pet, img_pet_old)
            t_car2 = time.time()
            if dist < 0.8:
                print("Ne Pohoge ", dist)
                cv2.imwrite(filename, img_pet)
        img_pet_old = img_pet
        # dets = cnn_face_detector(resc_frame, 1)
        # if len(dets) > 0:
        #    for i, d in enumerate(dets):
        #        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        #                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        #        x_left, y_top, x_right, y_bottom = normal_rect(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())
        #        face_img =resc_frame[y_top:y_bottom, x_left:x_right]
        #        filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S")+".jpg"
        #        cv2.imwrite(filename, face_img)
        #        # filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + "_resc.jpg"
        #        # cv2.imwrite(filename, resc_frame)
        #        # filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + "_rgb.jpg"
        #        # cv2.imwrite(filename, rgb_frame)
        #        filename = datetime.datetime.now().strftime("%d%m%Y__%H_%M_%S") + "_org.jpg"
        #        cv2.imwrite(filename, frame)
        #    t2 = time.time()
        #    print("Time:", (t2-t1))
        #    #rects = dlib.rectangles()
        #    #rects.extend([d.rect for d in dets])
        #    #win.clear_overlay()
        #    #win.set_image(resc_frame)
        #    #win.add_overlay(rects)
