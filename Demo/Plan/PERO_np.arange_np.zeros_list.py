import timeit

loop_times = 2 ** 16

for script in [
    "[0 for _ in range(n)]",
    "[i for i in range(n)]",
    "np.zeros(n, np.int)",
    "np.arange(n)",
]:
    print(timeit.repeat(stmt=script, setup="import numpy as np;n = 2 ** 9;",
                        repeat=2, number=loop_times))

"""
[2.0564617830023235, 1.795643180586791]
[1.6117533459143214, 1.5369785699837797]
[0.13760958652508481, 0.11909945438194303]
[0.12012791384374921, 0.12155749387735248]
"""