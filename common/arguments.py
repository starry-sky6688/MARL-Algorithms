import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='3', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='8m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    parser.add_argument('--alg', type=str, default='coma', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='the optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='the number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./model', help='the result directory of the policy')
    parser.add_argument('--learn', type=bool, default=False, help='whether to train the model')
    parser.add_argument('--threshold', type=int, default=19, help='the threshold to judge whether win')
    args = parser.parse_args()
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # the number of the epoch to train the agent
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 5

    # how often to save the model
    args.save_cycle = 50

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd and qmix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.0047
    args.min_epsilon = 0.03

    # the number of the epoch to train the agent
    args.n_epoch = 40000

    # the number of the episodes in one epoch
    args.n_episodes = 8

    # the number of the train steps in one epoch
    args.train_steps = 5

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(4e3)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200
    return args


