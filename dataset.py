"""Prepare dataset

This script extracts features from images, normalizes features and 
stores features to HDF5 file (dataset.h5). Samples will be shuffled.

Script saves fitted X_scaler and parameters to `params.pickle`.
"""

import argparse
import os
import pickle

from vt import colorspace
from vt import features
from vt import kitti
from vt import normalize

import h5py
import numpy as np
from sklearn.utils import shuffle


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python dataset.py')
    parser.add_argument('--features-color-space', type=str, default='YCrCb')
    parser.add_argument('--hist-bins', type=int, default=64)
    parser.add_argument('--spatial-size', type=int, default=32)
    parser.add_argument('--hog-orient', type=int, default=9)
    parser.add_argument('--hog-pix-per-cell', type=int, default=8)
    parser.add_argument('--hog-cell-per-block', type=int, default=2)
    parser.add_argument('--output', default='dataset.h5')
    parser.add_argument('--params-file', default='params.pickle')

    args = parser.parse_args()

    if args.features_color_space not in colorspace.SUPPORTED:
        print('unsupported color space: %s' % args.features_color_space)
        exit(1)

    def features_extractor(image):
        return features.extract(image,
                                hist_bins=args.hist_bins,
                                spatial_size=(args.spatial_size, args.spatial_size),
                                orient=args.hog_orient,
                                pix_per_cell=args.hog_pix_per_cell,
                                cell_per_block=args.hog_cell_per_block,
                                features_color_space=args.features_color_space)

    car_features = [features_extractor(img) for img in kitti.read('dataset/vehicles')]
    notcar_features = [features_extractor(img) for img in kitti.read('dataset/non-vehicles')]

    X, X_scaler = normalize.normalize(car_features, notcar_features)
    y = np.hstack((
        np.ones(len(car_features)),
        np.zeros(len(notcar_features)),
    ))

    X, y = shuffle(X, y)

    # save dataset
    if os.path.isfile(args.output):
        os.unlink(args.output)
    out = h5py.File(args.output)

    try:
        out.create_dataset('features', data=X)
        out.create_dataset('targets', data=y)
    finally:
        out.close()

    # save features parameters
    params = dict(
        scaler=X_scaler,
        features_color_space=args.features_color_space,
        hist_bins=args.hist_bins,
        spatial_size=(args.spatial_size, args.spatial_size),
        orient=args.hog_orient,
        pix_per_cell=args.hog_pix_per_cell,
        cell_per_block=args.hog_cell_per_block,
    )

    with open(args.params_file, 'wb') as params_file:
        pickle.dump(params, params_file)
