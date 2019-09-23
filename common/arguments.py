import argparse

"""
Here are the param for the training

"""


# TODO 把buffer里的avail_u分成avail_u和avail_u_next，这样让所有的项长度都为episode_limit
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='3', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='8m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--rnn_hidden_dim', type=int, default='64', help='the dimension of the hidden layer in RNN')
    parser.add_argument('--qmix_hidden_dim', type=int, default='32', help='the dimension of the hidden layer in qmix_net')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise epsilon')
    parser.add_argument('--epsilon', type=float, default=0.5, help='random epsilon for epsilon-greedy')
    parser.add_argument('--anneal_epsilon', type=float, default=0.0047, help='the anneal number of epsilon per epoch')
    parser.add_argument('--min_epsilon', type=float, default=0.03, help='the min epsilon')
    parser.add_argument('--batch_size', type=int, default=32, help='the sample batch size')
    parser.add_argument('--buffer_size', type=int, default=int(4e3), help='the number of the episodes can bestored in the buffer')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.0005, help='the learning rate of the critic')
    parser.add_argument('--n_episodes', type=int, default=8, help='the number of the episodes in one epoch')
    parser.add_argument('--n_epoch', type=int, default=5000, help='the number of the epoch to train the agent')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='the number of the epoch to evaluate the agent')
    parser.add_argument('--train_steps', type=int, default=5, help='how many times to train the agent in on epoch')
    parser.add_argument('--model_dir', type=str, default='./model', help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./model', help='the result directory of the policy')
    parser.add_argument('--save_cycle', type=int, default=500, help='how often to save the model, from the train_steps perspect')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target_net')
    parser.add_argument('--learn', type=bool, default=False, help='whether to train the model')
    parser.add_argument('--threshold', type=int, default=19, help='the threshold to judge whether win')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    args = parser.parse_args()

    return args

