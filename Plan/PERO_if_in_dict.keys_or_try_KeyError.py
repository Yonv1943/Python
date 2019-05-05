import time

'''performance analyse'''


def main():
    d = dict(zip(range(5431234), range(5431234)))
    d['N/A'] = 0
    timer = time.time()
    for i in range(12341234):
        # if i in d.keys():  # 1.8s
        #     d[i] += 1
        # else:
        #     d['N/A'] += 1

        try:  # 2.3s
            d[i] += 1
        except KeyError:
            d['N/A'] += 1
    print("||| TIME:", time.time() - timer)


if __name__ == '__main__':
    main()

pass
