"""Pipeline"""

import collections
import pickle

from vt import heatmap
from vt import subsampling
from vt import visual

import cv2
import numpy as np
import scipy.io


def draw_frameno(img, frameno):
    cv2.putText(img, '# ' + str(frameno), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)


class Pipeline:

    def __init__(self,
                 svc,
                 df_threshold=0.7,
                 threshold=30,
                 history_length=25,
                 params_file='params.pickle',
                 debug=False):

        with open(svc, 'rb') as svc_file:
            self.svc = pickle.load(svc_file)

        with open(params_file, 'rb') as params_file:
            dataset_params = pickle.load(params_file)

        self.X_scaler = dataset_params['scaler']

        self.extractor_params = dataset_params.copy()
        del self.extractor_params['scaler']

        # thresholds
        self.df_threshold = df_threshold
        self.threshold = threshold

        self.n = 1  # frames counter

        self.history_length = history_length
        self.history = collections.deque(maxlen=history_length)

        self.debug = debug

        # metrics
        self.windowsn = []  # number of hot windows
        self.heatmin = []
        self.heatmean = []
        self.heatmax = []
        self.rheatmin = []
        self.rheatmean = []
        self.rheatmax = []
        self.carn = []

    @property
    def heatmap(self):
        if not self.history:
            return None
        heatmap = np.zeros_like(self.history[0])
        for i, h in enumerate(self.history):
            w = float(i + 1) / float(self.history_length)
            heatmap = heatmap + h
        return heatmap

    def save_metrics(self, metrics_file):
        scipy.io.savemat(metrics_file, dict(
            windowsn=self.windowsn,
            heatmin=self.heatmin,
            heatmean=self.heatmean,
            heatmax=self.heatmax,
            rheatmin=self.rheatmin,
            rheatmean=self.rheatmean,
            rheatmax=self.rheatmax,
            carn=self.carn,
        ))

    def __call__(self, bgr_frame):
        print('frame #', self.n)
        self.n += 1

        ystart = 400
        ystop = 656

        hot_windows = []

        for scale in [4.0, 2.5, 2.0, 1.5]:
            ws = subsampling.find_cars(bgr_frame, ystart, ystop, scale, self.svc,
                                       self.X_scaler,
                                       df_threshold=self.df_threshold,
                                       **self.extractor_params)
            hot_windows.extend(ws)

        self.windowsn.append(len(hot_windows))

        if self.debug:
            bgr_frame = visual.draw_boxes(bgr_frame, hot_windows, (255, 0, 0))

        current_hm = heatmap.HeatMap(bgr_frame)
        current_hm.add_heat(hot_windows)

        # collect current heatmap metrics
        self.heatmin.append(current_hm.min)
        self.heatmean.append(current_hm.mean)
        self.heatmax.append(current_hm.max)

        self.history.append(current_hm.heatmap)

        hm = heatmap.HeatMap(bgr_frame, self.heatmap)

        # collect accumulated heat map metrics
        self.rheatmin.append(hm.min)
        self.rheatmean.append(hm.mean)
        self.rheatmax.append(hm.max)

        hm.apply_threshold(self.threshold)

        self.carn.append(hm.carn)
        hm.draw(bgr_frame)

        if self.debug:
            draw_frameno(bgr_frame, self.n)
        return bgr_frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('python pipeline.py')
    parser.add_argument('--svc', default='svc.pickle')
    parser.add_argument('--params-file', default='params.pickle')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--input-video', required=True)
    parser.add_argument('--output-video', default='output.avi')
    parser.add_argument('--threshold', default=30, type=int)
    parser.add_argument('--history-length', default=25, type=int)
    parser.add_argument('--metrics-file', default='metrics.mat')
    parser.add_argument('--df-threshold', default=.7, type=float)

    args = parser.parse_args()

    from udacitylib import video

    p = Pipeline(svc=args.svc,
                 params_file=args.params_file,
                 debug=args.debug,
                 df_threshold=args.df_threshold,
                 threshold=args.threshold,
                 history_length=args.history_length)

    video.convert(args.input_video, p, args.output_video)

    p.save_metrics(args.metrics_file)
