"""
    Functions taken from:
    https://github.com/Shikhargupta/computer-vision-techniques/blob/master/GaborFilter/gabor.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from mahotas.features import haralick
from skimage.feature import greycomatrix, greycoprops


# define gabor filter bank with different orientations and at different scales
def build_filters():
    """
    Constructs gabor filter to be used
    Returns
    -------
    filters:    gabor filter databank

    """
    filters = []
    ksize = 9
    # define the range for theta and nu
    for theta in np.arange(0, np.pi, np.pi / 6):
        for sigma in (1, 3):
            for frequency in np.arange(2, 11, 2):
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
                kern /= 1 * kern.sum()  # 1.5
                filters.append(kern)
    return filters


def process(img, filters, debug=False):
    """

    Parameters
    ----------
    img         color image
    filters     filter databank
    debug       view filter response

    Returns
    -------
    accum:      REsponse of the image to the databank

    """
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, -1, kern)
        np.maximum(accum, fimg, accum)

    if debug:
        plt.subplot(1, 2, 1)
        plt.imshow(filters)
        plt.subplot(1, 2, 2)
        plt.imshow(accum)
        plt.show()
    return accum


def gabor_features(imgg):
    """
    Calculate the gabor features of an image
    Parameters
    ----------
    imgg        image to process

    Returns
    -------
    features    gabor features of the image

    """
    if len(imgg.shape) == 2:
        channels = 2
        # img = cv2.cvtColor(imgg,cv2.COLOR_BGR2HSV)
        img = imgg
    else:
        channels = 3
        img = imgg
        # img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        # channels = 2

    # instantiating the filters:  This 2 ines must be outside to speed up
    filters = build_filters()
    f = np.asarray(filters)

    # initializing the feature vector
    # feat = np.zeros((len(f), 5 * channels), dtype=np.double)
    feat = None
    # calculating the local energy for each convolved image
    for j in range(len(f)):
        res = process(img, f[j])
        energy = np.mean(res * res, axis=(0, 1))
        mean = np.mean(res, axis=(0, 1))
        var = np.var(res, axis=(0, 1))
        rango = np.max(res, axis=(0, 1)) - np.min(res, axis=(0, 1))
        if channels == 3:
            moda = np.array((mode(res[:, :, 0], axis=None)[0][0],
                             mode(res[:, :, 1], axis=None)[0][0],
                             mode(res[:, :, 2], axis=None)[0][0]))
        else:
            moda = mode(res, axis=None)[0][0]
        temp = np.concatenate((mean, var, energy, rango, moda))

        if feat is None:
            feat0 = temp.reshape((1, -1))
            feat = np.repeat(feat0, len(f), axis=0)

        feat[j, :] = temp

    features = feat.flatten()
    # features matrix is the feature vector for the image

    return features
