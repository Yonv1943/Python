import os
import multiprocessing as mp

import cv2
import numpy as np

'''
2018-07-05 Yonv1943 show file images, via multiprocessing
2018-09-04 use multiprocessing for loading images
2018-09-05 add simplify
'''


def img_load(queue, queue_idx__img_paths):
    while True:
        idx, img_path = queue_idx__img_paths.get()
        img = cv2.imread(img_path)  # Disk IO
        queue.put((img, idx, img_path))


# def img_show_simplify(queue, window_name=''):
#     cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
#
#     while True:
#         img, idx, img_path = queue.get()
#         cv2.imshow(window_name, img)
#         cv2.waitKey(1)


def img_show(queue, window_name=''):  # check images and keep order
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    import bisect
    idx_previous = -1
    idxs = list()
    queue_gets = list()
    while True:
        queue_get = queue.get()
        idx = queue_get[1]
        insert = bisect.bisect(idxs, idx)  # keep order
        idxs.insert(insert, idx)
        queue_gets.insert(insert, queue_get)

        # print(idx_previous, idxs)
        while idxs and idxs[0] == idx_previous + 1:
            idx_previous = idxs.pop(0)
            img, idx, img_path = queue_gets.pop(0)
            if not isinstance(img, np.ndarray):  # check images
                os.remove(img_path)
                print("| Remove no image:", idx, img_path)
            elif not (img[-4:, -4:] - 128).any():  # download incomplete
                os.remove(img_path)
                print("| Remove incomplete image:", idx, img_path)
            else:
                cv2.imshow(window_name, img)
                cv2.waitKey(1)


def run():
    src_path = 'F:/url_get_image/ftp.nnvl.noaa.gov_GER_2018'
    img_paths = [os.path.join(src_path, f) for f in os.listdir(src_path) if f[-4:] == '.jpg']
    print("|Directory perpare to load:", src_path)
    print("|Number of images:", len(img_paths), img_paths[0])

    mp.set_start_method('spawn')

    queue_img = mp.Queue(8)
    queue_idx__img_path = mp.Queue(len(img_paths))
    [queue_idx__img_path.put(idx__img_path) for idx__img_path in enumerate(img_paths)]

    processes = list()
    processes.append(mp.Process(target=img_show, args=(queue_img,)), )
    processes.extend([mp.Process(target=img_load, args=(queue_img, queue_idx__img_path))
                      for _ in range(3)])

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    run()
