import multiprocessing as mp
import cv2
import time
import pickle
import gzip

"""
Source: Yonv1943 2019-05-04
https://github.com/Yonv1943/Python/upload/master/Demo/server_client_camera.py
https://zhuanlan.zhihu.com/p/64534116

Pickle EOFError: Ran out of input when recv from a socket - Antti Haapala
from multiprocessing.connection import Client
https://stackoverflow.com/a/24727097/9293137

How can I get the IP address of eth0 in Python? - jeremyjjbrown
s.getsockname()[0]
https://stackoverflow.com/a/30990617/9293137
"""


def client_img_put(q, name, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (name, pwd, ip, channel))
    is_opened = cap.read()[0]
    if is_opened:
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (name, pwd, ip, channel))
        print('DaHua')

    while is_opened:
        q.put(cap.read()[1]) if is_opened else None
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def client_img_get(q, window_name, host, port):
    # cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)

    from multiprocessing.connection import Client
    client = Client((host, port))

    '''init'''
    frame = q.get()
    shape = tuple([i // 3 for i in frame.shape[:2][::-1]])  # (1080P)

    times = 0
    time0 = time.time()
    while time.time() - time0 < 10:
        times += 1
        frame = q.get()

        frame = cv2.resize(frame, shape)
        frame = pickle.dumps(frame)
        # frame = gzip.compress(frame, compresslevel=1)
        client.send(frame)

        frame = client.recv()
        # frame = gzip.decompress(frame)
        frame = pickle.loads(frame)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
    print("fps:", times / 10)


def run_client(host, port):
    user_name, user_pwd, camera_ip = "admin", "admin123456", "172.20.114.26"

    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=2)
    processes = [
        mp.Process(target=client_img_put, args=(queue, user_name, user_pwd, camera_ip)),
        mp.Process(target=client_img_get, args=(queue, camera_ip, host, port)),
    ]

    [setattr(process, "daemon", True) for process in processes]  # process.daemon = True
    [process.start() for process in processes]
    [process.join() for process in processes]


def run_server(host, port):
    from multiprocessing.connection import Listener
    server_sock = Listener((host, port))
    print('Server Listening')

    conn = server_sock.accept()
    print('Server Accept')

    while True:
        frame = conn.recv()
        # frame = gzip.decompress(frame)
        frame = pickle.loads(frame)

        '''data = image_processing(data)'''
        # time.sleep(0.5)
        # cv2.imshow('ImgShow', data)
        # cv2.waitKey(1)

        frame = pickle.dumps(frame)
        # frame = gzip.compress(frame, compresslevel=1)
        conn.send(frame)


if __name__ == '__main__':
    server_host = '10.10.1.111'  # host = 'localhost'
    server_port = 32928  # if [Address already in use], use another port


    def get_ip_address(remote_server="8.8.8.8"):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((remote_server, 80))
        return s.getsockname()[0]


    if get_ip_address() == server_host:
        run_server(server_host, server_port)  # first, run this function only in server
    else:
        run_client(server_host, server_port)  # then, run this function only in client
    pass
