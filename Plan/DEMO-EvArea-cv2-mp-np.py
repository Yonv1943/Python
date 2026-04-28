"""Simple multiprocessing life-area simulation with OpenCV display."""

import itertools
import multiprocessing as mp
import time

import cv2
import numpy as np


ROOM = 4
SEAT = 3
SIDE = 48
QUEUE_SIZE = 4

INDEX_BLUE = 0
INDEX_GREEN = 1
INDEX_RED = 2
INDEX_END = 3
INDEX_LIFE = 0

EXPLORE = {
    0: np.array([1, 0]),
    1: np.array([0, 1]),
    2: np.array([-1, 0]),
    3: np.array([0, -1]),
}


def spread_seed(source, target):
    """Return updated source and target values."""
    target = source.copy()
    return source, target


def update_area(area, rng):
    """Update the simulation area by spreading values to neighbors."""
    for x_pos, y_pos in itertools.product(range(1, SIDE - 1), range(1, SIDE - 1)):
        if area[x_pos, y_pos, INDEX_LIFE] <= 0.0:
            continue

        next_x, next_y = EXPLORE[rng.integers(4)] + np.array([x_pos, y_pos])
        source = area[x_pos, y_pos].copy()
        target = area[next_x, next_y].copy()

        source, target = spread_seed(source, target)
        area[x_pos, y_pos] = source
        area[next_x, next_y] = target

    return area


def producer(queue_img, side):
    """Generate simulation frames and put them into the queue."""
    rng = np.random.default_rng()
    area = np.zeros((side, side, SEAT), dtype=float)

    seed_count = int(side ** 2 / ROOM ** 2)
    seed_positions = rng.integers(low=0, high=side, size=(seed_count, 2))

    for x_pos, y_pos in seed_positions:
        area[x_pos, y_pos] = rng.uniform(low=0.0, high=1.0, size=SEAT)

    while True:
        area = update_area(area, rng)
        if not queue_img.full():
            queue_img.put(area.copy())


def viewer(queue_img, window_name):
    """Display simulation frames with OpenCV."""
    cv2.namedWindow(window_name, flags=cv2.WINDOW_KEEPRATIO)
    previous_time = time.time()

    while True:
        area = queue_img.get()
        show = area[:, :, INDEX_BLUE:INDEX_END]

        max_value = np.max(show)
        if max_value > 0:
            show = (show * (255.0 / max_value)).astype(np.uint8)
        else:
            show = np.zeros_like(show, dtype=np.uint8)

        cv2.imshow(window_name, show)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        current_time = time.time()
        print(f"||| Ave time: {current_time - previous_time:.6f}")
        previous_time = current_time

    cv2.destroyAllWindows()


def main():
    """Start producer and viewer processes."""
    queue_img = mp.Queue(maxsize=QUEUE_SIZE)

    processes = [
        mp.Process(target=producer, args=(queue_img, SIDE)),
        mp.Process(target=viewer, args=(queue_img, "EvArea")),
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()