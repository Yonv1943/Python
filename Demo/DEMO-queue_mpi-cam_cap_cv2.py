import multiprocessing as mp
import cv2


def queue_img_put(q, name, pwd, ip, channel=3):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    is_opened, frame = cap.read()

    '''loop'''
    while is_opened:
        is_opened, frame = cap.read()
        q.put([is_opened, frame])


def queue_img_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    is_opened = True

    '''loop'''
    while is_opened:
        (is_opened, frame) = q.get()
        if q.qsize() < 4:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)


def main():
    user_name = "admin"
    user_pwd = "!QAZ2wsx3edc"

    camera_ip_l = [
        "192.168.1.164",
        "192.168.1.165",
        "192.168.1.166",
    ]

    mp.set_start_method(method='spawn')

    '''queue'''
    queue_img_l = [mp.Queue(maxsize=64) for _ in camera_ip_l]

    '''process'''
    process_io_2dl = [
        [mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)),
         mp.Process(target=queue_img_get, args=(queue, camera_ip))]
        for (queue, camera_ip) in zip(queue_img_l, camera_ip_l)
    ]

    '''start'''
    for process_l in process_io_2dl:
        for process in process_l:
            process.start()

    '''join'''
    for process_l in process_io_2dl:
        for process in process_l:
            process.join()


if __name__ == '__main__':
    main()
pass
