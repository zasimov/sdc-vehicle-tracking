"""Spatial Binning of Color"""

import cv2


def feature_vector_for_image(image, spatial_size):
    """Resize `image` to `spatial_size` and convert to a one dimensional feature fector"""
    resized = cv2.resize(image, spatial_size)
    return resized.ravel()

