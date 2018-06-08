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


Load the Hikvision WebCamera
Hikvision Manual Page 23
IP 192.168.1.64
http://192.168.1.64/doc/page/login.asp?_1523878379866
http port 80
USER admin
Universal Password

Can only use IE to open below website to the preview page
Can not use Chrome either FireFox, even Edge cannot open it
"http://192.168.1.64/doc/page/preview.asp"



Python：从subprocess运行的子进程中实时获取输出
https://blog.csdn.net/cnweike/article/details/73620250
"""


def video_capture_simplify(name, pwd, ip):
    """"""
    '''init'''
    video_pwd = "rtsp://%s:%s@%s//Streaming/Channels/1" % (name, pwd, ip)
    cap = cv2.VideoCapture(video_pwd)

    '''loop'''
    while True:
        success, frame = cap.read()
        cv2.imshow("WebCamera", frame)
        cv2.waitKey(1)


def video_capture(name, pwd, ip, channel_num=1):
    """"""
    '''init'''
    video_pwd = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel_num)
    window_name = "webCamera %s" % ip
    cap = cv2.VideoCapture(video_pwd)
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    print("||| Camera %s is opened: %s" % (ip, cap.isOpened()))

    is_opened = cap.isOpened()

    '''loop'''
    while is_opened:
        cap.read()  # skip frame
        success, frame = cap.read()
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == 13:  # ENTER, ord('\r') == 13
            is_opened = False


if __name__ == '__main__':
    user_name = "admin"
    user_pwd = "!QAZ2wsx3edc"
    camera_ip = "192.168.1.165"

    # video_capture_simplify(user_name, user_pwd, camera_ip)

    channel_number = 1
    video_capture(user_name, user_pwd, camera_ip, channel_number)

pass
