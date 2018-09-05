import cv2
import multiprocessing as mp
import os
import numpy as np

'''2018-07-05 Yonv1943 show file images, via multiprocessing'''


def queue_img_put(queue, img_paths):
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)  # Disk IO
        queue.put((img, i, img_path))


def queue_img_get(queue):
    window_name = ''
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    while True:
        img, i, img_path = queue.get()
        if not isinstance(img, np.ndarray):
            os.remove(img_path), print("| Remove no image:", i, img_path)
        elif not (img[-4:, -4:] - 128).any():  # download incomplete
            os.remove(img_path), print("| Remove incomplete image:", i, img_path)
        else:
            try:
                cv2.imshow(window_name, img), cv2.waitKey(1)
            except Exception as e:
                print("|", i, e)
            pass


def run():
    src_path = 'F:/url_get_image/ftp.nnvl.noaa.gov'
    img_paths = [os.path.join(src_path, f) for f in os.listdir(src_path) if f[-4:] == '.jpg'][13215:]
    print(len(img_paths), img_paths[0])

    mp.set_start_method('spawn')
    queue_img = mp.Queue(8)

    processes = [
        mp.Process(target=queue_img_put, args=(queue_img, img_paths)),
        mp.Process(target=queue_img_get, args=(queue_img,)),
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
