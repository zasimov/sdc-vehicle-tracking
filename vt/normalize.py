"""Normalize features"""

import numpy as np
from sklearn import preprocessing


def normalize(*features):
    """Combines and normalizes `features`
    
    Return one dimensional numpy array (float64)
    """
    X = np.vstack(features).astype(np.float64)
    X_scaler = preprocessing.StandardScaler().fit(X)
    return X_scaler.transform(X), X_scaler
