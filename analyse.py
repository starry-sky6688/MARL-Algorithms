import numpy as np
from policy.vdn import VDN
from policy.qmix import QMIX
from policy.coma import COMA
import os
import matplotlib.pyplot as plt
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args
from common.arguments import get_coma_args
from common.arguments import get_mixer_args


def plt_win_rate_mean():
    path = []
    alg_num = 6
    win_rates = [[] for _ in range(alg_num)]
    game_map = '3m'
    path.append('./result/vdn/' + game_map)
    path.append('./result/qmix/' + game_map)
    path.append('./result/qtran_base/' + game_map)
    path.append('./result/qtran_alt/' + game_map)
    path.append('./result/coma/' + game_map)
    path.append('./result/commnet_coma/' + game_map)
    num_run = 8
    for i in range(alg_num):
        for j in range(num_run):
            win_rates[i].append(np.load(path[i] + '/win_rates_{}.npy'.format(j)))
    win_rates = np.array(win_rates).mean(axis=1)
    new_win_rates = [[] for _ in range(alg_num)]
    average_cycle = 5
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
    new_win_rates[:, 0] = 0
    win_rates = new_win_rates


    plt.figure()
    plt.ylim(0, 1.0)
    plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='vdn')
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='qmix')
    plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qtran_base')
    plt.plot(range(len(win_rates[3])), win_rates[3], c='y', label='qtran_alt')
    plt.plot(range(len(win_rates[4])), win_rates[4], c='c', label='coma')
    plt.plot(range(len(win_rates[5])), win_rates[5], c='m', label='commnet_coma')
    plt.legend()
    plt.xlabel('epoch * 25')
    plt.ylabel('win_rate')
    plt.savefig('./result/overview.png')
    plt.show()

def plt_win_rate_single():
    path = []
    alg_num = 6
    win_rates = []
    path.append('./result/vdn.npy')
    path.append('./result/qmix.npy')
    path.append('./result/qtran_base.npy')
    path.append('./result/qtran_alt.npy')
    path.append('./result/coma.npy')
    path.append('./result/commnet_coma.npy')
    for i in range(alg_num):
        win_rates.append(np.load(path[i]))
    win_rates = np.array(win_rates)
    new_win_rates = [[] for _ in range(alg_num)]
    average_cycle = 5
    for i in range(alg_num):
        rate = 0
        time = 0
        for j in range(1000):
            rate += win_rates[i, j]
            time += 1
            if time % average_cycle == 0:
                new_win_rates[i].append(rate / average_cycle)
                time = 0
                rate = 0
    new_win_rates = np.array(new_win_rates)
    new_win_rates[:, 0] = 0
    win_rates = new_win_rates


    plt.figure()
    plt.ylim(0, 1.0)
    plt.plot(range(len(win_rates[0])), win_rates[0], c='b', label='vdn')
    plt.plot(range(len(win_rates[1])), win_rates[1], c='r', label='qmix')
    plt.plot(range(len(win_rates[2])), win_rates[2], c='g', label='qtran_base')
    plt.plot(range(len(win_rates[3])), win_rates[3], c='y', label='qtran_alt')
    plt.plot(range(len(win_rates[4])), win_rates[4], c='c', label='coma')
    plt.plot(range(len(win_rates[5])), win_rates[5], c='m', label='commnet_coma')
    plt.legend()
    plt.xlabel('epoch * 25')
    plt.ylabel('win_rate')
    plt.savefig('./result/best.png')
    plt.show()


def find_best_model(model_path, model_num):
    args = get_common_args()
    if args.alg == 'coma':
        args = get_coma_args(args)
        rnn_suffix = 'rnn_params.pkl'
        critic_fuffix = 'critic_params.pkl'
        policy = COMA
    elif args.alg == 'qmix':
        args = get_mixer_args(args)
        rnn_suffix = 'rnn_net_params.pkl'
        critic_fuffix = 'qmix_net_params.pkl'
        policy = QMIX
    elif args.alg == 'vdn':
        args = get_mixer_args(args)
        rnn_suffix = 'rnn_net_params.pkl'
        critic_fuffix = 'vdn_net_params.pkl'
        policy = VDN
    else:
        raise Exception("Not finished")
    env = StarCraft2Env(map_name=args.map,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version,
                        replay_dir=args.replay_dir)
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.evaluate_epoch = 100
    runner = Runner(env, args)
    max_win_rate = 0
    max_win_rate_idx = 0
    for num in range(model_num):
        critic_path = model_path + '/' + str(num) + '_' + critic_fuffix
        rnn_path = model_path + '/' + str(num) + '_' + rnn_suffix
        if os.path.exists(critic_path) and os.path.exists(rnn_path):
            os.rename(critic_path, model_path + '/' + critic_fuffix)
            os.rename(rnn_path, model_path + '/' + rnn_suffix)
            runner.agents.policy = policy(args)
            win_rate = runner.evaluate_sparse()
            if win_rate > max_win_rate:
                max_win_rate = win_rate
                max_win_rate_idx = num

            os.rename(model_path + '/' + critic_fuffix, critic_path)
            os.rename(model_path + '/' + rnn_suffix, rnn_path)
            print('The win rate of {} is  {}'.format(num, win_rate))
    print('The max win rate is {}, model index is {}'.format(max_win_rate, max_win_rate_idx))


if __name__ == '__main__':
    # plt_win_rate_mean()
    plt_win_rate_single()