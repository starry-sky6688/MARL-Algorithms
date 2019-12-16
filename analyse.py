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


def plt_win_rate():
    reward = np.load('./model/qtran_base/3m/episode_rewards.npy')
    rate = np.load('./model/qtran_base/3m/win_rates.npy')
    r_reward, r_rate = [], []
    a = 0
    num = 1
    end = 5500
    for i in range(end):
        a += reward[i]
        if i > 0 and i % num == 0:
            r_reward.append(a / num)
            a = 0
    a = 0
    for i in range(end):
        a += rate[i]
        if i > 0 and i % num == 0:
            r_rate.append(a / num)
            a = 0
    plt.figure()
    # plt.ylim(0, 1.0)
    # plt.plot(range(len(r_1)), r_1)
    # # plt.legend()
    # plt.xlabel('epoch')
    # plt.ylabel('win_rate')

    plt.subplot(2, 1, 1)
    plt.plot(range(len(r_rate)), r_rate)
    plt.xlabel('epoch')
    plt.ylabel('win_rate')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(r_reward)), r_reward)
    plt.xlabel('epoch')
    plt.ylabel('episode_rewards')

    plt.savefig('./model/qtran_base/3m/plt.png', format='png')
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
    plt_win_rate()
