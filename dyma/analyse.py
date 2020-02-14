import numpy as np
import matplotlib.pyplot as plt


def plt_win_rate_mean():
    path = []
    alg_num = 2
    win_rates = [[] for _ in range(alg_num)]
    path.append('./result/dyma/new_3m/no_trans')
    path.append('./result/dyma/new_3m/trans')

    num_run = 8
    for i in range(alg_num):
        for j in range(num_run):
            win_rates[i].append(np.load(path[i] + '/win_rates_{}.npy'.format(j)))
            # if i == 0:
            #     np.save('./result/dyma/new_3m/no_trans/win_rates_{}.npy'.format(j), win_rates[i][j])
            # else:
            #     np.save('./result/dyma/new_3m/trans/win_rates_{}.npy'.format(j), win_rates[i][j])
    win_rates = np.array(win_rates).mean(axis=1)
    new_win_rates = [[] for _ in range(alg_num)]
    average_cycle = 2
    for i in range(alg_num):
        rate = 0
        time = 0
        for j in range(len(win_rates[0])):
            rate += win_rates[i, j]
            time += 1
            if time % average_cycle == 0:
                new_win_rates[i].append(rate / average_cycle)
                time = 0
                rate = 0
    new_win_rates = np.array(new_win_rates)
    # new_win_rates[:, 0] = 0
    win_rates = new_win_rates


    plt.figure()
    plt.ylim(0, 1.0)
    plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='no_trans')
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='trans')
    plt.legend()
    plt.xlabel('epoch * 25')
    plt.ylabel('win_rate')
    plt.savefig('./result/compare.png', format='png')
    plt.show()


if __name__ == '__main__':
    plt_win_rate_mean()