import cv2
import time
import multiprocessing as mp
import numpy as np
import numpy.random as rd
import itertools

'''Yonv1943 2018-05-20 10:10:02'''


class Information(object):
    room = 4  # seed room
    seat = 3  # area seat
    side = 48  # area side
    size = 4  # vida multi-processing threading number

    explore = {  # vida, area (x, y)
        0: np.array([+1, +0]),
        1: np.array([+0, +1]),
        2: np.array([-1, +0]),
        3: np.array([+0, -1]),
    }
    '''index'''
    ib, ig, ir, di = 0, 1, 2, 3  # color: [blue, green, red]
    li = 0  # life
    tu = 3  # turn


i = Information()
rd.seed(int(time.time()))


def seed(m1, m2):
    m2 = m1
    return m1, m2


def vida(area):
    for (x1, y1) in itertools.product(range(1, i.side - 1), range(1, i.side - 1)):
        if area[x1, y1][i.li] <= 0.0:  # life == 0, continue
            continue

        (x2, y2) = i.explore[rd.randint(4)] + (x1, y1)
        m1, m2 = area[x1, y1], area[x2, y2]
        m1, m2 = seed(m1, m2)

        area[x1, y1], area[x2, y2] = m1, m2
    return area


def p_vida(q_img, side):
    area = np.zeros((side, side, i.seat), dtype=np.float)

    '''plant'''
    seed_num = int(side ** 2 / i.room ** 2)
    for (x, y) in rd.randint(low=0, high=i.side, size=(seed_num, 2)):
        area[x, y] = rd.uniform(low=0.0, high=1.0, size=i.seat)

    while True:
        area = vida(area)
        q_img.put(area)


def p_view(q_img, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_KEEPRATIO)
    timer0 = time.time()

    while True:
        area = q_img.get()
        show = np.array(area[:, :, i.ib:i.di])
        show = np.array(show * (255.0 / np.max(show)), dtype=np.uint8)
        cv2.imshow(window_name, show)
        cv2.waitKey(1)

        timer1 = time.time()
        print("||| Ave time:", timer1 - timer0)
        timer0 = timer1

def main():
    mp.set_start_method('spawn')
    queue_img = mp.Queue(maxsize=4)

    process_l = [
        mp.Process(target=p_vida, args=(queue_img, i.side)),
        mp.Process(target=p_view, args=(queue_img, 'EvArea')),
    ]

    [p.start() for p in process_l]
    [p.join() for p in process_l]


if __name__ == '__main__':
    main()
pass
