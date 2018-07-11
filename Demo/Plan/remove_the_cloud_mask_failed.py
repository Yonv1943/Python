import cv2
import os
import numpy as np


def backup():
    cv2.namedWindow('', cv2.WINDOW_KEEPRATIO)

    src_path = "F:/ftp.nnvl.noaa.gov"
    for img in [os.path.join(src_path, f) for f in os.listdir(src_path) if f[-4:] == '.jpg']:
        img = cv2.imread(img)
        cv2.imshow('', img), cv2.waitKey(1)


def get_cloud_mask(org, k=0.7):
    min_rgb = np.vectorize(lambda b, g, r: min(b, g, r))(org[:, :, 0], org[:, :, 1], org[:, :, 2])
    min_rgb.astype(np.float)
    min_rgb = min_rgb / 256.0

    mat = ((1.0 + k) * min_rgb - k)
    mat = np.vectorize(lambda x: x if x > 0.0 else 0.0)(mat)
    show = org * mat[:, :, np.newaxis].repeat(3, axis=2)
    return show


def get_cloud_mask1(org, j=3, k=3):
    min_rgb = show = np.vectorize(lambda b, g, r: min(b, g, r))(org[:, :, 0], org[:, :, 1], org[:, :, 2])
    min_rgb.astype(np.float)
    min_rgb = min_rgb / 256.0

    std_rgb = np.std(org, axis=2)
    std_rgb.astype(np.float)
    std_rgb = std_rgb / 256.0

    cv2.imshow('', org)
    while True:
        wait_key = cv2.waitKey(1)
        if wait_key != -1:
            if wait_key == 119:
                j *= 1.05
            elif wait_key == 115:
                j *= 0.95
            elif wait_key == 100:
                k *= 1.05
            elif wait_key == 97:
                k *= 0.95
            print("j, k: %.2f | %.2f | %3d" % (j, k, wait_key))

            mat = min_rgb * j - std_rgb * k

            # for mat in [min_rgb, std_rgb, mat]:
            #     print("np.range():", np.min(mat), np.max(mat))

            mat = np.minimum(mat, 1)
            mat = np.maximum(mat, 0)

            temp = show * mat
            temp = temp[:, :, np.newaxis].repeat(3, axis=2)
            cv2.imshow('', temp.astype(np.uint8))
    return show


def run():
    cv2.namedWindow('', cv2.WINDOW_KEEPRATIO)

    org = cv2.imread('test.jpg')
    org = cv2.resize(org, (org.shape[0] // 2, org.shape[1] // 2))

    show = get_cloud_mask1(org)

    # cv2.imshow('', org), cv2.waitKey(2345)
    cv2.imshow('', show.astype(np.uint8)), cv2.waitKey(4576)
