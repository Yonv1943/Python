import cv2


def video_capture_simplify(name, pwd, ip):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/1" % (name, pwd, ip))

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
        success, frame = cap.read()  # If frame is read correctly, it will be True.
        # cap.read()  # You could use this way to skip frame

        cv2.imshow(window_name, frame) if success else None
        is_opened = False if cv2.waitKey(1) == 8 else True
        # press ENTER to quit, cv2.waitKey(1) == 13 == ord('\r')
        # press BackSpace to quit, cv2.waitKey(1) == 8 == ord('\b')

    cap.release()
    cv2.destroyWindow(window_name)


def run():
    user_name = "admin"
    user_pwd = "!QAZ2wsx3edc"
    camera_ip = "192.168.1.169"

    # video_capture_simplify(user_name, user_pwd, camera_ip)
    video_capture(user_name, user_pwd, camera_ip, channel_num=1)


if __name__ == '__main__':
    run()
