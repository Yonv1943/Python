import multiprocessing as mp
import cv2


def queue_img_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))

    while True:
        is_opened, frame = cap.read()
        q.put(frame) if is_opened else None
        q.get() if q.qsize() > 1 else None


def queue_img_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)

    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run():
    user_name, user_pwd = "admin", "!QAZ2wsx3edc"

    camera_ip_l = [
        "192.168.1.169",
        "192.168.1.170",
    ]

    '''init'''
    mp.set_start_method(method='spawn')

    img_queues = [mp.Queue(maxsize=2) for _ in camera_ip_l]  # queue

    io_process_2d = [
        [mp.Process(target=queue_img_put, args=(queue, user_name, user_pwd, camera_ip)),
         mp.Process(target=queue_img_get, args=(queue, camera_ip))]
        for (queue, camera_ip) in zip(img_queues, camera_ip_l)
    ]

    '''start'''
    for process_l in io_process_2d:
        for process in process_l:
            process.daemon = True
    [[process.start() for process in process_l] for process_l in io_process_2d]
    [[process.join() for process in process_l] for process_l in io_process_2d]


if __name__ == '__main__':
    run()
pass
