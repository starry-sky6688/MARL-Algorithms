import torch
import torch.nn as nn
import torch.nn.functional as f


# input obs of all agents，output probability distribution of all agents
class CommNet(nn.Module):
    def __init__(self, input_shape, args):
        super(CommNet, self).__init__()
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  # 对所有agent的obs解码
        self.f_obs = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 每个agent根据自己的obs编码得到hidden_state，用于记忆之前的obs
        self.f_comm = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 用于通信
        self.decoding = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):
        # 先对obs编码
        obs_encoding = torch.sigmoid(self.encoding(obs))  # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # 第一次经过f_obs得到h
        h_out = self.f_obs(obs_encoding, h_in)

        for k in range(self.args.k):  # 通信self.args.k次
            if k == 0:  # 初始化c为0
                h = h_out
                c = torch.zeros_like(h)
            else:
                # 把h转化出n_agents维度用于通信
                h = h.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

                # 对于每个agent，其他agent的h相加
                # 先让最后一维包含所有agent的h
                c = h.reshape(-1, 1, self.args.n_agents * self.args.rnn_hidden_dim)
                c = c.repeat(1, self.args.n_agents, 1)  # 此时每个agent都有了所有agent的h
                # 把每个agent自己的h置0
                mask = (1 - torch.eye(self.args.n_agents))  # th.eye（）生成一个二维对角矩阵
                mask = mask.view(-1, 1).repeat(1, self.args.rnn_hidden_dim).view(self.args.n_agents, -1)  # (n_agents, n_agents * rnn_hidden_dim))
                if self.args.cuda:
                    mask = mask.cuda()
                c = c * mask.unsqueeze(0)
                # 因为现在所有agent的h都在最后一维，不能直接加。所以先扩展一维，相加后再去掉
                c = c.reshape(-1, self.args.n_agents, self.args.n_agents, self.args.rnn_hidden_dim)
                c = c.mean(dim=-2)  # (episode_num * max_episode_len, n_agents, rnn_hidden_dim)
                h = h.reshape(-1, self.args.rnn_hidden_dim)
                c = c.reshape(-1, self.args.rnn_hidden_dim)
            h = self.f_comm(c, h)

        # 通信结束后计算每个agent的所有动作的权重，概率在agent.py中选择动作时计算
        weights = self.decoding(h)

        return weights, h_out

