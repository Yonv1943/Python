import time

'''performance analyse'''

l = range(2 ** 22)

time0 = time.time()
a = [i for i in l if i % 2 == 0]
print("%.4f" % (time.time() - time0), len(a))

time0 = time.time()
a = [i for i in l if i % 2 == 0]
print("%.4f" % (time.time() - time0), len(a))

time0 = time.time()
a = [i for i in filter(lambda x: bool(x % 2 == 0), l)]
print("%.4f" % (time.time() - time0), len(a))

time0 = time.time()
a = list(filter(lambda x: bool(x % 2 == 0), l))
print("%.4f" % (time.time() - time0), len(a))

"""
0.5584 2097152
0.5123 2097152 --------- [for if]
1.6361 2097152
1.5931 2097152
"""
