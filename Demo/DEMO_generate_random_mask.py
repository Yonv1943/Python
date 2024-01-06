import cv2
import numpy as np

# Source: LearningToPaint - hzwer
# Modify: Yonv1943 Github
# DEMO_generate_random_mask.py

def brush_bezier_curve(thetas, width1=128):
    # paras = np.random.rand(128, 10)
    width2 = width1 * 2
    thetas[:, 0:4] *= width2
    thetas[:, 4:6] = thetas[:, 0:2] + (thetas[:, 2:4] - thetas[:, 0:2]) * thetas[:, 4:6]
    thetas[:, 6:8] = thetas[:, 6:8] * (width2 // 8) + 2
    thetas[:, 8:10] *= 255

    res = np.empty((thetas.shape[0], width1, width1), dtype=np.uint8)
    for idx, para in enumerate(thetas):
        x0, y0, x2, y2, x1, y1, z0, z2, w0, w2 = para

        canvas = np.zeros([width2, width2], dtype=np.uint8)
        gap = width1 // 2
        for p in range(gap):
            p /= gap
            q = 1 - p

            pp = p * p
            qq = q * q
            pq2 = p * q * 2

            x = int(pp * x2 + pq2 * x1 + qq * x0)
            y = int(pp * y2 + pq2 * y1 + qq * y0)
            z = int(p * z2 + q * z0)
            w = int(p * w2 + q * w0)

            cv2.circle(canvas, (y, x), z, w, -1)
        res[idx] = cv2.resize(canvas, dsize=(width1, width1))

    return res

def generate_random_mask(thetas, width2=128):
    # paras = np.random.rand(128, 8)
    thetas[:, 0:4] *= width2
    thetas[:, 4:6] = thetas[:, 0:2] + (thetas[:, 2:4] - thetas[:, 0:2]) * thetas[:, 4:6]
    thetas[:, 6:8] = thetas[:, 6:8] * (width2 // 8) + 2

    canvas = np.ones([width2, width2], dtype=np.uint8)
    gap = width2 // 2
    for idx, para in enumerate(thetas):
        x0, y0, x2, y2, x1, y1, z0, z2 = para
        for p in range(gap):
            p /= gap
            q = 1 - p

            pp = p * p
            qq = q * q
            pq2 = p * q * 2

            x = int(pp * x2 + pq2 * x1 + qq * x0)
            y = int(pp * y2 + pq2 * y1 + qq * y0)
            z = int(p * z2 + q * z0)

            cv2.circle(canvas, (y, x), z, 0, -1)

    return canvas

if __name__ == '__main__':
    WIDTH = 128
    Mask = generate_random_mask(np.random.rand(2, 8), WIDTH)
    Mask = Mask[:, :, np.newaxis]

    Img = cv2.imread('image/lisa.png')
    cv2.imshow('Random Mask', Img * Mask)
    cv2.waitKey(12340)

    for image in brush_bezier_curve(np.random.rand(8, 10), WIDTH):
        cv2.imshow('Random Mask', image)
        cv2.waitKey(234)
