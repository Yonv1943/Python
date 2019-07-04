import time
import multiprocessing as mp
import cv2

"""
Source: Yonv1943 2018-06-17
https://github.com/Yonv1943/Python
https://zhuanlan.zhihu.com/p/38136322
OpenCV official demo
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
海康、大华IpCamera RTSP地址和格式（原创，旧版）- 2014年08月12日 23:01:18 xiejiashu
rtsp_path_hikvison = "rtsp://%s:%s@%s/h265/ch%s/main/av_stream" % (user, pwd, ip, channel)
rtsp_path_dahua = "rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel)
https://blog.csdn.net/xiejiashu/article/details/38523437
最新（2017）海康摄像机、NVR、流媒体服务器、回放取流RTSP地址规则说明 - 2017年05月13日 10:51:46 xiejiashu
rtsp_path_hikvison = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel)
https://blog.csdn.net/xiejiashu/article/details/71786187
"""


def image_put(q, user, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_opencv_camera():
    user, pwd, ip, channel = "admin", "admin123456", "172.20.114.26", 1

    cap_path = 0  # local camera (e.g. the front camera of laptop)
    # cap_path = 'video.avi'  # the path of video file
    # cap_path = "rtsp://%s:%s@%s/h264/ch%s/main/av_stream" % (user, pwd, ip, channel)  # HIKIVISION old version 2015
    # cap_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel)  # HIKIVISION new version 2017
    # cap_path = "rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel)  # dahua

    cap = cv2.VideoCapture(cap_path)

    while cap.isOpened():
        is_opened, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(1000)
    cap.release()


def run_single_camera():
    # user_name, user_pwd, camera_ip = "admin", "admin123456", "172.20.114.196"
    user_name, user_pwd, camera_ip = "admin", "admin123456", "[fe80::3aaf:29ff:fed3:d260]"

    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)),
                 mp.Process(target=image_get, args=(queue, camera_ip))]

    [process.start() for process in processes]
    [process.join() for process in processes]


def run_multi_camera():
    user_name, user_pwd = "admin", "admin123456"
    camera_ip_l = [
        "172.20.114.196",  # ipv4
        "[fe80::3aaf:29ff:fed3:d260]",  # ipv6
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


def image_collect(queue_list, camera_ip_l):
    import numpy as np

    """show in single opencv-imshow window"""
    window_name = "%s_and_so_no" % camera_ip_l[0]
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        imgs = [q.get() for q in queue_list]
        imgs = np.concatenate(imgs, axis=1)
        cv2.imshow(window_name, imgs)
        cv2.waitKey(1)

    # """show in multiple opencv-imshow windows"""
    # [cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    #  for window_name in camera_ip_l]
    # while True:
    #     for window_name, q in zip(camera_ip_l, queue_list):
    #         cv2.imshow(window_name, q.get())
    #         cv2.waitKey(1)


def run_multi_camera_in_a_window():
    user_name, user_pwd = "admin", "admin123456"
    camera_ip_l = [
        "172.20.114.196",  # ipv4
        "[fe80::3aaf:29ff:fed3:d260]",  # ipv6
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = [mp.Process(target=image_collect, args=(queues, camera_ip_l))]
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))

    for process in processes:
        process.daemon = True  # setattr(process, 'deamon', True)
        process.start()
    for process in processes:
        process.join()


def run():
    # run_opencv_camera()  # slow, with only 1 thread
    run_single_camera()  # quick, with 2 threads
    # run_multi_camera() # with 1 + n threads
    # run_multi_camera_in_a_window()  # with 1 + n threads
    pass


if __name__ == '__main__':
    run()
