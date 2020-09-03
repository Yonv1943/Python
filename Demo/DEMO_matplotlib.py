# import os
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

"""DEMO of matplotlib, by Github YonV1943 
"""


def run():
    data = """
    90639	15659	229016	10948	3502186	540924	617798	56957	613851	109118
    94754	14401	243284	11886	3339180	673668	479573	45890	591279	99243
    58708	8737	178966	19048	6346521	1068038	893739	561353	913392	212345
    68646	12639	181829	20301	6751437	1228376	991798	565571	1361534	259998
    67307	29807	195128	20305	5373321	967151	1548356	696006	2070184	374223
    77695	32106	192996	23117	6233739	1059212	1445726	787749	2315160	332135
    72848	15149	212149	16699	5099217	669202	1279871	302160	1854453	250945
    """
    data_x = "LunarLanderContinuous-v2		BipedalWalker-v3		BipedalWalkerHardCore-v3		Ant-v1		Minitaur-v1"
    data_y = """
    ISAC+TC+SU+DP
    ISAC+TC+SU 
    ISAC+TC+SN
    ISAC+SC+SN
    IAC+SC+SN+DP
    IAC+SC+SN 
    IAC+TC+SN
    """
    total_width = 0.8

    data_x = data_x.split()
    data_y = [i_str[4:] for i_str in data_y.split('\n')[1:-1]]
    data = [i_str.split() for i_str in data.split('\n')[1:-1]]

    ary = np.array(data, dtype=np.int)
    ary = ary.reshape((len(data_y), len(data_x), 2))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # [3:]

    grep_id = np.array((0, 1))
    title = 'ISAC and IAC simple task'
    # grep_id = np.array((3, 4))
    # title = 'ISAC and IAC normal task'
    # grep_id = np.array((2, ))
    # title = 'ISAC and IAC hard task'
    data_x = np.array(data_x)[grep_id]
    ary = ary[:, grep_id]
    colors = np.array(colors)[grep_id]

    labels = data_x
    n_label = len(labels)

    n_bars = len(data_y)  # Number of bars per group
    bar_width = total_width / n_bars  # The width of a single bar

    bars = []
    bars_width = (n_label + 1) * bar_width
    fig, ax = plt.subplots()

    # print(';', len(data_x))
    # print(';', len(data_y))
    # print(';', ary.shape)

    for i, name in enumerate(data_y):
        means = ary[i, :, 0]
        errors = ary[i, :, 1]

        x_offset = i * bars_width  # The offset in j direction of that bar

        for j in range(n_label):
            loc = j * bar_width + x_offset
            bar = ax.bar(loc, means[j], yerr=errors[j],
                         width=bar_width, color=colors[j % len(colors)])
            if i == 0:
                bars.append(bar)

    ax.legend(bars, labels, loc='upper center')

    x_loc = np.arange(n_bars) * bars_width + (bars_width / 2 - bar_width)
    x_tricks = data_y
    plt.xticks(x_loc, x_tricks, rotation=20)
    plt.title(title)
    plt.grid()
    plt.gcf().subplots_adjust(bottom=0.2)
    # plt.show()
    save_path = f'comparison_target_reward_{title}.pdf'
    plt.savefig(save_path, dpi=200)
    print(save_path)


def plot__multi_error_bars(ary_avg, ary_std=None, labels0=None, labels1=None, title='multi_error_bars'):
    """
    labels0 = ['x-axis0', 'x-axis1', 'x-axis2', 'x-axis3']
    labels1 = ['legend0', 'legend1', 'legend2', 'legend3']
    ary_avg = np.random.rand(len(labels0), len(labels1))
    ary_std = np.random.rand(*ary_avg.shape) * 0.25  # None  #

    plot__multi_error_bars(ary_avg, ary_std, labels0, labels1)
    """
    if ary_std is None:
        ary_std = np.empty_like(ary_avg)
        ary_std[:, :] = None

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # [3:]

    lab0_len = len(labels0)  # Number of bars per group
    lab1_len = len(labels1)

    bar_width = 1 / lab0_len  # The width of a single bar

    bars = []
    bars_width = (lab1_len + 1) * bar_width
    fig, ax = plt.subplots()

    for i in range(lab0_len):
        avg = ary_avg[i, :]
        std = ary_std[i, :]

        x_offset = i * bars_width  # The offset in j direction of that bar

        for j in range(lab1_len):
            x1_loc = j * bar_width + x_offset
            bar = ax.bar(x1_loc, avg[j], yerr=std[j],
                         width=bar_width, color=colors[j % len(colors)])
            if i == 0:
                bars.append(bar)

    ax.legend(bars, labels1, loc='upper right')

    '''if the name of x-axis is too long, adjust the rotation and bottom'''
    x0_loc = np.arange(lab0_len) * bars_width - bar_width + bars_width / 2
    plt.xticks(x0_loc, labels0, rotation=15)
    plt.gcf().subplots_adjust(bottom=0.1)

    plt.title(title)
    plt.grid()

    plt.show()
    # save_path = 'multi_error_bars.pdf'
    # plt.savefig(save_path, dpi=200)
    # print(save_path)


def plot__error_std(ys, xs=None, k=8):
    """
    xs = np.linspace(0, 2, 64)
    ys = np.sin(xs) + rd.normal(0, 0.1, size=xs.shape[0])

    plot__error_plot(ys, xs, k=8)
    """
    if xs is None:
        xs = np.arange(ys.shape[0])

    ys_pad = np.pad(ys, pad_width=(k, 0), mode='edge')
    ys_avg = list()
    ys_std = list()
    for i in range(len(ys)):
        ys_part = ys_pad[i:i + k]
        ys_avg.append(ys_part.mean())
        ys_std.append(ys_part.std())

    plt.plot(xs, ys, color='royalblue')

    plt.plot(xs, ys_avg, color='lightcoral')
    ys_avg = np.array(ys_avg)
    ys_std = np.array(ys_std)
    plt.fill_between(xs, ys_avg - ys_std, ys_avg + ys_std, facecolor='lightcoral', alpha=0.3)
    plt.show()


def plot__error_plot_round(ys, xs=None, k=8):  # 2020-09-03
    """
    xs = np.linspace(0, 2, 64)
    ys = np.sin(xs)
    ys[rd.randint(64, size=8)] = 0
    plot__error_plot_round(ys, xs, k=8)
    """

    if xs is None:
        xs = np.arange(ys.shape[0])

    ys_pad = np.pad(ys, pad_width=(k // 2, k // 2), mode='edge')
    ys_avg = list()
    ys_std1 = list()
    ys_std2 = list()
    for i in range(len(ys)):
        ys_part = ys_pad[i:i + k]
        avg = ys_part.mean()
        ys_avg.append(avg)
        ys_std1.append((ys_part[ys_part > avg] - avg).mean())
        ys_std2.append((ys_part[ys_part <= avg] - avg).mean())

    # if is_padding:
    #     plt.plot(xs[:-k//2], ys[:-k//2], color='royalblue')
    # else:
    plt.plot(xs, ys, color='royalblue')

    plt.plot(xs, ys_avg, color='lightcoral')
    ys_avg = np.array(ys_avg)
    ys_std1 = np.array(ys_std1)
    ys_std2 = np.array(ys_std2)
    plt.fill_between(xs, ys_avg + ys_std1, ys_avg + ys_std2, facecolor='lightcoral', alpha=0.3)
    plt.show()


def run_demo():
    # labels0 = ['x-axis0', 'x-axis1', 'x-axis2', 'x-axis3']
    # labels1 = ['legend0', 'legend1', 'legend2', 'legend3']
    # ary_avg = np.random.rand(len(labels0), len(labels1))
    # ary_std = np.random.rand(*ary_avg.shape) * 0.25  # None  #
    #
    # plot__multi_error_bars(ary_avg, ary_std, labels0, labels1)

    xs = np.linspace(0, 2, 64)
    ys = np.sin(xs)
    ys[rd.randint(64, size=8)] = 0

    plot__error_plot_round(ys, xs, k=8)


if __name__ == '__main__':
    run_demo()
