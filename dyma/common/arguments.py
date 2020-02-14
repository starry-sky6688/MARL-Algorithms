import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='3', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    # The alternative algorithms are vdn、coma、qmix、qtran_base、qtran_alt and commnet_coma
    parser.add_argument('--alg', type=str, default='dyma', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='the optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='the number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='the result directory of the policy')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')
    parser.add_argument('--threshold', type=int, default=19, help='the threshold to judge whether win')
    args = parser.parse_args()
    return args


def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 2500

    # the number of the episodes in one epoch
    args.n_episodes = 8

    # the number of the train steps in one epoch
    args.train_steps = 8  # qtran:8

    # # how often to evaluate
    args.evaluate_cycle = 5

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 2000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10
    return args



