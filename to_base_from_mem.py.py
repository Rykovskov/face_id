#!/usr/bin/python3

import cv2
import dlib
import psycopg2
import datetime
import sys
import traceback
import numpy as np
import time
import _pickle as pickle
from scipy.spatial import distance
import os

critical_face_distance=0.5
directory = '/mnt/tmpfs'
size_pic_with=1920
size_pic_high=1080

conn = psycopg2.connect(dbname='an_photo', user='max',password='FfrGhBjvW5', host='144.76.219.163')
sql_insert_photo = """insert into in_photo (dt_photo, id_object, photo, id_face_descr) values (%s,%s,%s,%s)  RETURNING id_in_photo;"""
sql_insert_face_desc = """insert into face_descriptor (descriptor) values (%s) RETURNING id_face_descriptor;"""

#predictor_path = "shape_predictor_5_face_landmarks.dat"
predictor_path = "/home/max/project/shape_predictor_68_face_landmarks.dat"
facerec = dlib.face_recognition_model_v1('/home/max/project/dlib_face_recognition_resnet_model_v1.dat')
sp = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

files = os.listdir(directory) 
for file in files:
    try:
        filename = "/mnt/tmpfs/"+file
        cur = conn.cursor()   
        temp_array = []
        #Получили картинку
        t1=time.time()
        frame = cv2.imread(filename)
        dets = detector(frame, 1)
        s_images=[]
        i=0
        for k, d in enumerate(dets):
            x_face1=d.left()-40
            x_face2=d.right()+40
            y_face1=d.top()-60
            y_face2=d.bottom()+60                   
            if x_face1<0:
               x_face1=0
            if y_face1<0:
               y_face1=0
            if x_face2>size_pic_with:
               x_face2=size_pic_with
            if y_face2>size_pic_high:
               y_face2=size_pic_high                       
            //s_images.append(frame[y_face1:y_face2, x_face1:x_face2])
            //i=i+1
        num_faces = len(dets)
        if num_faces > 0:
           faces = dlib.full_object_detections()
           for detection in dets:
               faces.append(sp(frame, detection))
               # Get the aligned face images
               # Optionally: 
               #images = dlib.get_face_chips(rgb_image, faces, size=320, padding=0.25)
               images = dlib.get_face_chips(frame, faces, size=640, padding=0.25)
               for image in images:
                   try:
                       cv2.imshow('facedetect', image)
                       #img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                       dets1 = detector(image, 1)
                       for k1, d1 in enumerate(dets1):
                           shape1 = sp(image, d1)
                       face_descriptor1 = facerec.compute_face_descriptor(image, shape1)
                       now = datetime.datetime.now()
                       try:
                           out_dump1 = pickle.dumps(face_descriptor1,1)
                           cur.execute(sql_insert_face_desc, (psycopg2.Binary(out_dump1),))
                           id_of_new_row_descr = cur.fetchone()[0]
                           cv2.imwrite("77677.jpg", image)
                           mypic=open("77677.jpg",'rb').read()
                           cur.execute(sql_insert_photo, (now, 1, psycopg2.Binary(mypic), id_of_new_row_descr))
                           id_of_new_row = cur.fetchone()[0]
                           conn.commit()
                           #print("Save image complete!!")
                       except psycopg2.Error as e:
                           cur.close()
                           #conn.close()
                           print(e.pgerror)
                           print(e.diag.message_detail)
                           print('Ошибка:\n', traceback.format_exc())    
                   except Exception as e:
                        cur.close()
                        #conn.close()
                        print('Ошибка:\n', traceback.format_exc())
        t2=time.time()
        print("Time:",(t2-t1))
        os.remove(filename)
    except Exception as e:
        cur.close()
        cv2.destroyAllWindows()
        print(e.pgerror)
        print(e.diag.message_detail)
        print('Ошибка:\n', traceback.format_exc())
        cv2.destroyAllWindows()
        sys.exit(0)    
