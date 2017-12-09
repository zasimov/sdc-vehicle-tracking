"""Calculate color features"""


import cv2
import numpy as np
from matplotlib import pyplot as plt


def bin_centers(hist):
    """Return bin centers
    
    `hist` is a result of `np.histogram`
    """
    bin_edges = hist[1]
    if len(bin_edges) == 0:
        return bin_edges
    return (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2


class HistFeatures:

    def __init__(self, bgr_image, bins):
        r = bgr_image[:, :, 2]
        g = bgr_image[:, :, 1]
        b = bgr_image[:, :, 0]

        # rhist (bhist, ghist) is a tuple with counters and ranges
        # rhist[0]-rhist[1] is a first range (excluding the last element)
        self.rhist = np.histogram(r, bins=bins, range=(0, 256))
        self.ghist = np.histogram(g, bins=bins, range=(0, 256))
        self.bhist = np.histogram(b, bins=bins, range=(0, 256))

        # bin centers are the same for r, g, and b
        self.bin_centers = bin_centers(self.rhist)

    @property
    def features(self):
        return np.concatenate((self.rhist[0], self.ghist[0], self.bhist[0]))

    def plot(self, figsize=(12, 3)):
        # Plot a figure with all three bar charts
        figure = plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.bar(self.bin_centers, self.rhist[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(self.bin_centers, self.ghist[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(self.bin_centers, self.bhist[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        return figure


def extract(bgr_image, bins):
    hist = HistFeatures(bgr_image, bins=bins)
    return hist.features
