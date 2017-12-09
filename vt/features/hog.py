"""Calculate HOG features"""

import numpy as np
from skimage import feature as hog_feature

from vt import colorspace


def visualise_hog_features(gray_img, orient, pix_per_cell, cell_per_block):
    _, hog_image = hog_feature.hog(
        gray_img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=False,
        block_norm='L2-Hys',
        visualise=True,
        feature_vector=False,
    )
    return hog_image


def get_hog_features(gray_img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    """Calculate HOG features"""
    features = hog_feature.hog(
        gray_img,
        orientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=False,
        block_norm='L2-Hys',
        visualise=False,
        feature_vector=feature_vec,
    )
    return features


def extract(feature_image, orient, pix_per_cell, cell_per_block):
    """Extract HOG features for each channel
    
    Return one dimensional numpy array
    """
    hog_features = []  # per channel HOG features

    for channel in range(feature_image.shape[2]):
        fv = get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block)
        hog_features.append(fv)

    return np.ravel(hog_features)
