import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('Demo_read_plot_from_image.png')

img = img[:, :, 0]
print(img.shape)

xs = np.linspace(0, 317, 9)
print(xs)

ys = list()
for x in xs:
    ns = img[:, int(x)]
    i=0
    for i, n in enumerate(ns):
        if n < 200:
            break
    ys.append(i)

ys = (252-np.array(ys)) / 252 * 35
np.save('n8k4.npy', ys)
plt.plot(xs, ys)
plt.ylim(0)
plt.show()

# cv2.imshow('', img)
# cv2.waitKey()