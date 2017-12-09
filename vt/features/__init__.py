"""Features extractor"""

from vt import colorspace
from vt.features import histogram
from vt.features import hog
from vt.features import spatial_binning

import numpy as np


def extract(bgr_image, hist_bins, spatial_size, orient, pix_per_cell, cell_per_block, features_color_space):
    """Extract Raw features from `bgr_image`
    
    I use this function with
      hist_bins=32
      spatial_size=(32, 32)
      orient=9
      pix_per_cell=8
      cell_per_block=2
      features_color_space='YCrCb'
      
    Arguments don't have default values for predictability.
    
    Return features vector (numpy array):
      * hist
      * spatial
      * hog
    """
    feature_image = colorspace.change(bgr_image, 'BGR', features_color_space)
    hist = histogram.extract(feature_image, hist_bins)
    spatial = spatial_binning.feature_vector_for_image(feature_image, spatial_size)
    hog_features = hog.extract(feature_image, orient, pix_per_cell, cell_per_block)
    return np.concatenate([hist, spatial, hog_features])
