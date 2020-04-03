from __future__ import division
import sys
import os
import argparse
import logging
import numpy as np
import cv2
from scipy.spatial import distance
import dlib

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


def calculate_SIFT(img):
  # Find the keypoints and descriptors using SIFT features
  kp, des = sift.detectAndCompute(img,None)
  return kp, des

def CompareHash(hash1, hash2):
    l = len(hash1)
    i = 0
    count = 0
    while i < l:
        if hash1[i] != hash2[i]:
            count = count + 1
        i = i + 1
    if count > 1000:
         count = 1000

    return (1000-count)/1000

def Get_Difference(img1, img2):
    kp_img1, des_img1 = calculate_SIFT(img1)
    kp_img2, des_img2 = calculate_SIFT(img2)
    return CompareHash(hash1, hash2)