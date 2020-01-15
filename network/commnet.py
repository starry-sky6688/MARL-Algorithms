import torch
import torch.nn as nn
import torch.nn.functional as f


# 输入所有agent的obs，输出所有agent的动作
class CommNet(nn.Module):
    def __init__(self, input_shape, args):
        super(CommNet, self).__init__()
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  # 对所有agent的obs解码
        self.f = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 每个agent根据自己的obs编码得到hidden_state
        self.decoding = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def init_hidden(self):
        # make hidden states on same device as model
        # 返回一个1*rnn_hidden_dim的0向量
        return self.encoding.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        # 先对obs编码
        obs_encoding = torch.sigmoid(self.encoding(obs))  # .reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # 第一次经过f得到h
        h = self.f(obs_encoding, h_in)

        for k in range(self.args.k):  # 通信self.args.k次

            # 把h转化出n_agents维度用于通信
            h = h.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)

            # 对于每个agent，其他agent的h相加
            # 先让最后一维包含所有agent的h
            c = h.reshape(-1, 1, self.args.n_agents * self.args.rnn_hidden_dim)
            c = c.repeat(1, self.args.n_agents, 1)  # 此时每个agent都有了所有agent的h
            # 把每个agent自己的h置0
            mask = (1 - torch.eye(self.args.n_agents))  # th.eye（）生成一个二维对角矩阵
            mask = mask.view(-1, 1).repeat(1, self.args.rnn_hidden_dim).view(self.args.n_agents, -1)  # (n_agents, n_agents * rnn_hidden_dim))
            c = c * mask.unsqueeze(0)
            # 因为现在所有agent的h都在最后一维，不能直接加。所以先扩展一维，相加后再去掉
            c = c.reshape(-1, self.args.n_agents, self.args.n_agents, self.args.rnn_hidden_dim)
            c = c.mean(dim=-2)  # (episode_num * max_episode_len, n_agents, rnn_hidden_dim)

            h_in = h.reshape(-1, self.args.rnn_hidden_dim)
            c = c.reshape(-1, self.args.rnn_hidden_dim)
            h = self.f(c, h_in)

        # 通信结束后计算每个agent的所有动作的权重，概率在agent.py中选择动作时计算
        weights = self.decoding(h)

        return weights, h

