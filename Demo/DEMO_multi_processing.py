import time
import numpy as np
import multiprocessing as mp

"""An Tutorial of multi-processing (a Python built-in library)
"""


def function_pipe1(conn):
    p_id = 1
    print(p_id, 0)
    time.sleep(1)

    conn.send(np.ones(1))
    print(p_id, 'send1')
    ary = conn.recv()
    print(p_id, 'recv1', ary.shape)
    conn.send(np.ones(1))
    print(p_id, 'send2')

    time.sleep(3)


def function_pipe2(conn):
    p_id = '\t\t2'
    print(p_id, 0)
    time.sleep(1)

    conn.send(np.ones(2))
    print(p_id, 'send1')

    ary = conn.recv()
    print(p_id, 'recv1', ary.shape)
    ary = conn.recv()
    print(p_id, 'recv2', ary.shape)

    time.sleep(3)


def func1(i):
    time.sleep(1)
    print(f'id {i}')


def func2(args):  # multiple parameters (arguments)
    # x, y = args
    x = args[0]  # write in this way, easier to locate errors
    y = args[1]  # write in this way, easier to locate errors

    time.sleep(1)  # pretend it is a time-consuming operation
    return x - y


def run__pool():  # main process
    from multiprocessing import Pool

    cpu_worker_num = 3
    process_args = [(1, 1), (9, 9), (4, 4), (3, 3), ]

    print(f'| inputs:  {process_args}')
    start_time = time.time()
    with Pool(cpu_worker_num) as p:
        outputs = p.map(func2, process_args)
    print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

    '''Another way (I don't recommend)
    Using 'functions.partial'. See https://stackoverflow.com/a/25553970/9293137
    from functools import partial
    # from functools import partial
    # pool.map(partial(f, a, b), iterable)
    '''


def run__process():  # mp: multiprocessing
    from multiprocessing import Process
    process = [Process(target=func1, args=(1,)),
               Process(target=func1, args=(2,)), ]
    [p.start() for p in process]
    [p.join() for p in process]


def run__pipe():
    conn1, conn2 = mp.Pipe()

    process = [mp.Process(target=function_pipe1, args=(conn1,)),
               mp.Process(target=function_pipe2, args=(conn2,)), ]
    [p.start() for p in process]
    [p.join() for p in process]


if __name__ == '__main__':  # it is necessary to write main process in "if __name__ == '__main__'"
    # run__process()
    run__pool()
    # demo__pool()
