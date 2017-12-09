"""Module to work with Journal"""

import csv
import datetime
import os


class Journal:

    HEADER = ['date',
              'features_color_space',
              'hist_bins',
              'spatial_size',
              'orient',
              'pix_per_cell',
              'cell_per_block',
              'train_time',
              'model',
              'random_state',
              'C',
              'accuracy']

    def __init__(self, filename):
        write_header = False if os.path.isfile(filename) else True
        self.fd = open(filename, 'a')
        self.writer = csv.DictWriter(self.fd, fieldnames=self.HEADER)
        if write_header:
            self.writer.writeheader()

    def close(self):
        if self.fd:
            self.fd.close()
            self.fd = None

    def write(self, **kwargs):
        if kwargs.keys() - set(self.HEADER):
            raise TypeError('extra fields in kwargs: %s' % (kwargs.keys() - set(self.HEADER)))
        self.writer.writerow(kwargs)


def log(file_name, params, train_time, model, random_state, C, accuracy):
    row = params.copy()
    del row['scaler']
    row['date'] = datetime.datetime.now()
    row['train_time'] = train_time
    row['model'] = model
    row['random_state'] = random_state
    row['C'] = C
    row['accuracy'] = accuracy
    j = Journal(file_name)
    try:
        j.write(**row)
    finally:
        j.close()
