"""Heat map"""

import cv2
import numpy as np

from scipy.ndimage.measurements import label


def search_windows(img, windows, predictor, window_size=(64, 64)):
    """Predict for each window
    
    `predictor` is a function (image) -> [1, 0]
    """
    on_windows = []
    for window in windows:
        #  Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], window_size)
        prediction = predictor(test_img)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


class HeatMap:

    def __init__(self, image, heatmap=None):
        if heatmap is not None:
            self._heatmap = heatmap
        else:
            self._heatmap = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint32)

    @property
    def heatmap(self):
        return self._heatmap

    @property
    def min(self):
        return np.min(self._heatmap)

    @property
    def mean(self):
        return np.mean(self._heatmap)

    @property
    def max(self):
        return np.max(self._heatmap)

    def add_heat(self, hot_windows):
        for box in hot_windows:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self._heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def apply_threshold(self, threshold):
        # Zero out pixels below the threshold
        self._heatmap[self._heatmap <= threshold] = 0
        # Return thresholded map
        return self._heatmap

    @property
    def carn(self):
        _, carn = label(self._heatmap)
        return carn

    def draw(self, img, color=(0, 0, 255), thinkness=6):
        labels, carn = label(self._heatmap)
        for car_number in range(1, carn + 1):
            nonzero = (labels == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color, thinkness)
        return img
