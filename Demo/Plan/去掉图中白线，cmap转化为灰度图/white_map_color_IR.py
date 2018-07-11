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
                cv2.imshow(window_name, img), cv2.waitKey(5)
            except Exception as e:
                print("|", i, e)
            pass


def run1():
    src_path = 'F:/url_get_image/ftp.nnvl.noaa.gov_color_IR_2018'
    img_paths = [os.path.join(src_path, f) for f in os.listdir(src_path) if f[-4:] == '.jpg']
    print(len(img_paths), img_paths[0])

    mp.set_start_method('spawn')
    queue_img = mp.Queue(4)

    processes = [
        mp.Process(target=queue_img_put, args=(queue_img, img_paths)),
        mp.Process(target=queue_img_get, args=(queue_img,)),
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


def get_white_line_eliminate_map():
    window_name = ''
    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

    line_img = 'white_line_LMBWV2018-01-25-171633.jpg'
    line_img = cv2.imread(line_img)  # Disk IO
    line_img = line_img[:, :, 1]
    # line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    line_img = cv2.threshold(line_img, 127, 255, cv2.THRESH_BINARY)[1]

    # shift_pts = []
    # for i in range(line_img.shape[0]):
    #     for j in range(line_img.shape[1]):
    #         if line_img[i, j] == 255:
    #             shift_pts.append((i, j, 0, 0))
    # np.save('test', np.array(shift_pts))
    shift_pts = np.load('test.npy')
    print(2222222, shift_pts.shape)

    line_set = set([(i, j) for i, j in shift_pts[:, :2]])

    for id, (i, j, x, y) in enumerate(shift_pts):
        if x != 0 or y != 0:
            if (i, j - 1) not in line_set:
                shift_pts[id, 2:] = (i, j - 1)
            elif (i, j + 1) not in line_set:
                shift_pts[id, 2:] = (i, j + 1)
            elif (i - 1, j) not in line_set:
                shift_pts[id, 2:] = (i - 1, j)
            elif (i + 1, j) not in line_set:
                shift_pts[id, 2:] = (i + 1, j)
            else:
                while (i, j) in line_set:
                    i -= 1
                else:
                    shift_pts[id, 2:] = (i, j)

    np.save('test', np.array(shift_pts))
    shift_pts = np.load('test.npy')
    print(2222222, shift_pts.shape)

    test_img = cv2.imread('test.jpg')
    show_img = np.copy(test_img)
    for i, j, x, y in shift_pts:
        if x != 0 and y != 0:
            try:
                # show_img[i, j] = test_img[x, y]
                line_img[i, j] = line_img[x, y]
            except IndexError:
                print(i, j, x, y)

    shift_pts = np.load('test.npy')
    print(2222222, shift_pts.shape)

    cv2.imwrite('out.jpg', line_img)
    # show_imgs = [
    #     # line_img,
    #     test_img,
    #     show_img,
    # ]
    # for img in show_imgs:
    #     cv2.imshow(window_name, img), cv2.waitKey(5432)


def switch_ir_to_gray(img, map_pts):
    # img = cv2.imread('test.jpg')
    # map_pts = np.load('white_line_eliminate_map.npy')
    for i, j, x, y in map_pts:
        img[i, j] = img[x, y]

    out = img[:, :, 1]
    out = np.maximum(out, 60)
    out = np.minimum(out, 187)

    green = 60 - out
    gray = out - 60

    switch = np.abs(img[:, :, 1] - img[:, :, 0])
    switch = np.minimum(switch, 1)

    out = green * switch + gray * (1 - switch)
    out = out.astype(np.uint8)

    # window_name = ''
    # cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
    # cv2.imshow(window_name, out), cv2.waitKey()
    return out


if __name__ == '__main__':
    run()
