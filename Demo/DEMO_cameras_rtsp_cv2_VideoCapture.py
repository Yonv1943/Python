import cv2

"""
Ubuntu下使用Python-opencv获取海康威视RTSP码流教程
https://blog.csdn.net/hui3909/article/details/53435379

XviD-1.3.5
http://www.linuxfromscratch.org/blfs/view/svn/multimedia/xvid.html
Download (HTTP): http://downloads.xvid.org/downloads/xvidcore-1.3.5.tar.gz
Download size: 804 KB


Ubuntu16.04下安装FFmpeg（超简单版）
sudo add-apt-repository ppa:djcj/hybrid
sudo apt-get update
sudo apt-get install ffmpeg


Can only use IE to open below website to the HIKVISION Web Camera preview page
Can not use Chrome either FireFox, even Edge cannot open it
"http://192.168.1.64/doc/page/preview.asp"


Python：从subprocess运行的子进程中实时获取输出
https://blog.csdn.net/cnweike/article/details/73620250
"""


def video_capture_simplify(name, pwd, ip):
    video_pwd = "rtsp://%s:%s@%s//Streaming/Channels/1" % (name, pwd, ip)
    cap = cv2.VideoCapture(video_pwd)

    while True:  # loop
        success, frame = cap.read()
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)


def video_capture(name, pwd, ip, channel_num=1):
    """
    Source: Getting Started with Videos - opencv-python
    https://docs.opencv.org/3.4.1/dd/d43/tutorial_py_video_display.html
    Modify: Yonv1943
    """
    video_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel_num)
    window_name = "CameraIP: %s" % ip

    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    cap = cv2.VideoCapture(video_path)
    is_opened = cap.isOpened()
    print("||| CameraIP %s is opened: %s" % (ip, is_opened))

    while is_opened:  # loop
        # cap.read()  # You could use this way to skip frame
        success, frame = cap.read()  # If frame is read correctly, it will be True.
        cv2.imshow(window_name, frame) if success else None
        is_opened = False if cv2.waitKey(1) == 8 else True
        # press ENTER to quit, cv2.waitKey(1) == 13 == ord('\r')
        # press BackSpace to quit, cv2.waitKey(1) == 8 == ord('\b')

    cv2.destroyWindow(window_name)


def run():
    user_name = "admin"
    user_pwd = "!QAZ2wsx3edc"
    camera_ip = "192.168.1.169"

    # video_capture_simplify(user_name, user_pwd, camera_ip)
    video_capture(user_name, user_pwd, camera_ip, channel_num=1)


if __name__ == '__main__':
    run()
