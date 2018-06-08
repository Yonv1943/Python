import multiprocessing as mp
import time
import cv2
import os


def video_capture(q, name, pwd, ip):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/3" % (name, pwd, ip))
    is_opened, frame = cap.read()

    '''loop'''
    while is_opened:
        is_opened, frame = cap.read()
        q.put([is_opened, frame])


def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    is_opened = True

    save_file = "video_capture_%s" % window_name
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    else:
        print("FileExist:", save_file)

    gaps = 0
    while is_opened:
        (is_opened, frame) = q.get()

        if q.qsize() < 4:
            # time.sleep(1.0)

            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

            '''save image'''
            # if gaps > 4 and False:
            if gaps > 4:
                print("video_capture_%s/%s.jpg" % (window_name, time.time()))
                cv2.imwrite("%s/%s.jpg" % (save_file, time.time()), frame)
                gaps = 0
            gaps += 1


def main():
    user_name = "admin"
    user_pwd = "!QAZ2wsx3edc"

    camera_164_ip = "192.168.1.164"
    camera_165_ip = "192.168.1.165"
    camera_166_ip = "192.168.1.166"

    mp.set_start_method(method='spawn')

    '''queue'''
    queue_image_164 = mp.Queue(maxsize=64)
    queue_image_165 = mp.Queue(maxsize=64)
    queue_image_166 = mp.Queue(maxsize=64)

    '''process'''
    proce_camera_164 = mp.Process(target=video_capture, args=(queue_image_164, user_name, user_pwd, camera_164_ip))
    proce_camera_165 = mp.Process(target=video_capture, args=(queue_image_165, user_name, user_pwd, camera_165_ip))
    proce_camera_166 = mp.Process(target=video_capture, args=(queue_image_166, user_name, user_pwd, camera_166_ip))

    proce_image_get_164 = mp.Process(target=image_get, args=(queue_image_164, camera_164_ip))
    proce_image_get_165 = mp.Process(target=image_get, args=(queue_image_165, camera_165_ip))
    proce_image_get_166 = mp.Process(target=image_get, args=(queue_image_166, camera_166_ip))

    '''start'''
    proce_camera_164.start()
    proce_camera_165.start()
    proce_camera_166.start()

    proce_image_get_164.start()
    proce_image_get_165.start()
    proce_image_get_166.start()

    '''join'''
    proce_camera_164.join()
    proce_camera_165.join()
    proce_camera_166.join()

    proce_image_get_164.join()
    proce_image_get_165.join()
    proce_image_get_166.join()


if __name__ == '__main__':
    main()
pass
