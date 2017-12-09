"""Predictor is a chain: extract features -> scale -> classify"""

import numpy as np


class Predictor:

    def __init__(self, features_extractor, scaler, classifier):
        self.features_extractor = features_extractor
        self.scaler = scaler
        self.classifier = classifier

    def __call__(self, image):
        """Return 1 or 0"""
        features = self.features_extractor(image)
        test_features = self.scaler.transform(np.array(features).reshape(1, -1))
        return self.classifier.predict(test_features)
