import multiprocessing as mp
import numpy as np
import time

"""
https://stackoverflow.com/questions/10721915/
shared-memory-objects-in-multiprocessing/10724332#10724332

https://stackoverflow.com/questions/5549190/
is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156
"""


def convert_mp_to_np(mp_array):
    return np.ctypeslib.as_array(mp_array.get_obj())


def worker0(mp_ary, pipe0):
    import time

    time.sleep(1)

    ary = convert_mp_to_np(mp_ary)
    print(0, ary)

    pipe0.send(True)
    # pipe1.recv()

    mp_ary[:] = np.ones(3, dtype=np.float32)


def worker1(mp_ary, pipe1):
    # pipe0.send(True)
    pipe1.recv()  # wait worker0

    ary = convert_mp_to_np(mp_ary)
    print(1, ary)


def run_mp():

    np_type = np.ctypeslib.as_ctypes_type(np.float32)
    mp_ary = mp.Array(np_type, 3)
    pipe0, pipe1 = mp.Pipe()

    process = [mp.Process(target=worker0, args=(mp_ary, pipe0)),
               mp.Process(target=worker1, args=(mp_ary, pipe1))]
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':
    run_mp()


