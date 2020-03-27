from __future__ import division
import sys
import os
import argparse
import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()


def calculate_SIFT(img):
  # Find the keypoints and descriptors using SIFT features
  kp, des = sift.detectAndCompute(img,None)
  return kp, des


def knn_match(des1, des2, nn_ratio=0.7):
    # FLANN parameters
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match features from each image
    matches = flann.knnMatch(des1, des2, k=2)

    # store only the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < nn_ratio * n.distance:
            good.append(m)

    return good


# calculate the angle with the horizontal
def angle_horizontal(v):
    return -np.arctan2(v[1],v[0])