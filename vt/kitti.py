"""Functions to load KITTI dataset to memory (RAM)"""

import glob
import os

import cv2


def listdir(folder, ext='*.png'):
    """Read image names one-by-one"""
    for subfolder in os.listdir(folder):
        cwd = os.path.join(folder, subfolder)
        if not os.path.isdir(cwd):
            continue
        for img_name in glob.glob1(cwd, ext):
            yield os.path.join(cwd, img_name)


def read(folder, ext='*.png'):
    """Read images one-by-one"""
    for img_name in listdir(folder, ext):
        yield cv2.imread(img_name)
