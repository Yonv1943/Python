import cv2
import os
import numpy as np

img_src = input('image_src_path:')
img_dst = input('image_dst_path:')
rotation_time = int(input('anti-clockwise rotation 90 degree\n how many times?'))

img_name_l = [os.path.join(img_src, f) for f in os.listdir(img_src)]
for img_name in img_name_l:
    img = cv2.imread(os.path.join(img_src, img_name))
    img = np.rot90(img, k=rotation_time)  # anti-clockwise rotation
    cv2.imwrite(os.path.join(img_dst, img_name), img)
