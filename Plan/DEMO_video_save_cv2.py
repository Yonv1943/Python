import cv2
import numpy as np

'''
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

http://docs.opencv.org/3.1.0/d7/d9e/tutorial_video_write.html
https://github.com/ContinuumIO/anaconda-issues/issues/223

2018-09-06 10:47:25 Yonv1943

'''

win_name = 'cv2'
cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)
cap = cv2.VideoCapture('./datasets/bilibili-av22642627-zebra.mp4')
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
# out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))


i = 0
img = np.random.randint(0, 255, (480, 640, 3)).astype('uint8')
while cap.isOpened():
    is_opened, frame = cap.read()

    if 833 < i < 1444:
        out.write(img)
        out.write(cv2.resize(frame, (640, 480)).astype('uint8'))

    cv2.imshow(win_name, frame)
    print(i)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print('break')
