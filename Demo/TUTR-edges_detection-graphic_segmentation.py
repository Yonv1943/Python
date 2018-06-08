import cv2
import numpy as np

'''REFER: https://hub.packtpub.com/opencv-detecting-edges-lines-shapes/'''


def draw_contours(img, cnts):
    img = np.copy(img)
    img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    return img


def draw_box_circle(img, cnts):
    img = np.copy(img)
    for cnt in cnts:
        # find bounding box coordinates
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

        # find minimum area
        rect = cv2.minAreaRect(cnt)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)  # green

        # calculate center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img, center, radius, (0, 0, 255), 2)  # red
    return img


def draw_ploygen_approx_hull(img, cnts):
    img = np.copy(img)
    for cnt in cnts:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)

        hull = cv2.convexHull(cnt)
        cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)
    return img


if __name__ == '__main__':
    image = cv2.imread('test.jpg')
    cv2.imshow("contours", image)
    cv2.waitKey(1234)

    # ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.Canny(image, 222, 333)
    cv2.imshow("contours", thresh)
    cv2.waitKey(1234)

    thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("hierarchy:", hierarchy)

    cv2.imshow("contours", draw_contours(image, contours))
    cv2.waitKey(1234)
    cv2.imshow("contours", draw_ploygen_approx_hull(image, contours))
    cv2.waitKey(1234)
    cv2.imshow("contours", draw_box_circle(image, contours))
    cv2.waitKey(1234)
