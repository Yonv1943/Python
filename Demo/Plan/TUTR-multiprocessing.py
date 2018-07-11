import os
import time
import multiprocessing as mp
from multiprocessing import Pool, Process, Pipe, Lock, Value, Array, Manager


def func(x):
    print("||| Sleep time:", x)
    time.sleep(x)
    print("|||", __name__, os.getppid(), os.getpid())
    return x ** 2


def foo(q):
    q.put('hello')


def func_conn(conn):
    conn.send([42, None])
    conn.close()


def func_lock(l, i):
    l.acquire()
    try:
        sleep_time = i * 5 % 3
        time.sleep(sleep_time)
        print("|||", i, sleep_time)
    finally:
        l.release()
        pass


def func_memo(n, a):
    n.value += 1
    for i in range(len(a)):
        a[i] = a[i] + 1


def func_mana(d, l):
    d[1] = '1'
    d['two'] = 2
    d[0.25] = None
    l.reverse()


timer = time.time()
if __name__ == '__main__':
    pass

    # with Manager() as manager:
    #     d = manager.dict()
    #     l = manager.list(range(8))
    #
    #     p = Process(target=func_mana, args=(d, l))
    #     p.start()
    #     p.join()
    #
    #     print(d)
    #     print(l)

    # num = Value(typecode_or_type='d')
    # num.value = 42
    # arr = Array(typecode_or_type='i', size_or_initializer=range(10))
    # p1 = Process(target=func_memo, args=(num, arr))
    # p2 = Process(target=func_memo, args=(num, arr))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    #
    # print(num.value)
    # print(arr[:])

    with Pool(2) as p0:
        print(p0.map(func, [3, 1, 2]))

    # p1 = Process(target=func, args=(1,))
    # p1.start()
    # print(111111111111)
    # p1.join()

    # mp.set_start_method(method='spawn')
    # q = mp.Queue()
    # p = mp.Process(target=foo, args=(q,))
    # p.start()
    # print(q.get())
    # p.join()

    # ctx = mp.get_context('spawn')
    # q = ctx.Queue()
    # p = ctx.Process(target=foo, args=(q,))
    # p.start()
    # print(q.get())
    # p.join()

    # parent_conn, child_conn = Pipe()
    # p = Process(target=func_conn, args=(child_conn,))
    # p.start()
    # print(parent_conn.recv())
    # p.join()

    # lock = Lock()
    # for i in range(10):
    #     Process(target=func_lock, args=(lock, i)).start()

print("||| Total Time:", time.time() - timer)
pass

"""
Learning how to use [multiprocessing] in Python. (Official)
https://docs.python.org/3.6/library/multiprocessing.html

multiprocessing Process join run, 李皮筋的技术博客
https://www.cnblogs.com/lipijin/p/3709903.html
"""