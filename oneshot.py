"""Run classifier for one image"""

import glob
import pipeline

import cv2
from matplotlib import pyplot

if __name__ == '__main__':

    images = glob.glob('test_images/*.jpg')

    r = len(images)
    c = 2
    n = 1

    pyplot.figure(figsize=(11, 10))

    for image_name in images:
        img = cv2.imread(image_name)

        pyplot.subplot(r, c, n)
        pyplot.imshow(img)
        n += 1

        p = pipeline.Pipeline(svc='svc-100.pickle',
                              df_threshold=.7,
                              threshold=25,
                              history_length=25,
                              debug=True)
        out = p(img)
        pyplot.subplot(r, c, n)
        pyplot.imshow(out)
        n += 1

    pyplot.tight_layout()
    pyplot.savefig('output_images/test_images.png')
    pyplot.show()

