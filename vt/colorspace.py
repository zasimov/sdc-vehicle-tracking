"""Helpers to change Color Space"""

import cv2
import numpy as np


conv = dict()
conv['RGB'] = dict()
conv['RGB']['RGB'] = None
conv['RGB']['BGR'] = cv2.COLOR_RGB2BGR
conv['RGB']['HSV'] = cv2.COLOR_RGB2HSV
conv['RGB']['LUV'] = cv2.COLOR_RGB2LUV
conv['RGB']['HLS'] = cv2.COLOR_RGB2HLS
conv['RGB']['YUV'] = cv2.COLOR_RGB2YUV
conv['RGB']['YCrCb'] = cv2.cv2.COLOR_RGB2YCrCb

conv['BGR'] = dict()
conv['BGR']['BGR'] = None
conv['BGR']['RGB'] = cv2.COLOR_BGR2RGB
conv['BGR']['HSV'] = cv2.COLOR_BGR2HSV
conv['BGR']['LUV'] = cv2.COLOR_BGR2LUV
conv['BGR']['HLS'] = cv2.COLOR_BGR2HLS
conv['BGR']['YUV'] = cv2.COLOR_BGR2YUV
conv['BGR']['YCrCb'] = cv2.cv2.COLOR_BGR2YCrCb


SUPPORTED = {'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'}


def _get_cv2_color_space(src, dst):
    targets = conv.get(src)
    if not targets:
        return None
    return targets.get(dst)


def change(image, src, dst):
    cv2_color_space = _get_cv2_color_space(src, dst)
    if cv2_color_space is None:
        return np.copy(image)
    return cv2.cvtColor(image, cv2_color_space)
