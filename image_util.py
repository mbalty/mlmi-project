import sys

sys.path.append('/usr/local/lib/python3.5/site-packages')

import cv2 as cv
import numpy as np


def padd_image(img, edge):
    h = img.shape[0]
    w = img.shape[1]
    if h > w:
        r = float(edge) / h
        h = edge
        w = int(w * r)

        resized = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)
        hb = int(h / 2)
        wb = int(h - w / 2)
        mir = cv.copyMakeBorder(resized, hb, hb, wb, wb, cv.BORDER_REFLECT)
    else:
        r = float(edge) / w
        w = edge
        h = int(h * r)
        resized = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)

        wb = int(w / 2)
        hb = int(w - h / 2)
        mir = cv.copyMakeBorder(resized, hb, hb, wb, wb, cv.BORDER_REFLECT)
    return mir


def crop_center(img, w, h):
    w = int(w / 2)
    h = int(h / 2)
    center = (int(img.shape[:2][0] / 2.), int(img.shape[:2][1] / 2.))
    return img[center[0] - h:center[0] + h, center[1] - w:center[1] + h]


def augment_images(img, edge, aug_num=20):
    img = padd_image(img, edge)
    augmented = [crop_center(img, edge, edge)]
    (h, w) = img.shape[:2]

    center = (w / 2, h / 2)
    for i in range(aug_num):
        M = cv.getRotationMatrix2D(center, np.random.randint(0, high=360), 1.0)
        rotated = cv.warpAffine(img, M, (w, h))
        augmented.append(crop_center(rotated, edge, edge))
    return augmented


def preprocess_image(img, edge):
    img = padd_image(img, edge)
    return crop_center(img, edge, edge)


