import torch.nn as nn
import torch
import torch.nn.functional as F


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents代表使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim代表网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.qmix_hidden_dim)的矩阵
        self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
        # 经过hyper_w2得到(经验条数, 1)的矩阵
        self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  # q_values的shape为(32, 60, 5)，states的shape为(32, 60, 120)
        # 传入的q_values是三维的，shape为(episode个数, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # q_values的shape转化为(32 * 60=1920, 1, 5)
        states = states.reshape(-1, self.args.state_shape)  # (1920, 120)

        w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
        return q_total
