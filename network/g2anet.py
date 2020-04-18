import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


# 输入所有agent的obs，输出所有agent的动作概率分布
class G2ANet(nn.Module):
    def __init__(self, input_shape, args):
        super(G2ANet, self).__init__()

        # Encoding
        self.encoding = nn.Linear(input_shape, args.rnn_hidden_dim)  # 对所有agent的obs解码
        self.h = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # 每个agent根据自己的obs编码得到hidden_state，用于记忆之前的obs

        # Hard
        # GRU输入[[h_i,h_1],[h_i,h_2],...[h_i,h_n]]与[0,...,0]，输出[[h_1],[h_2],...,[h_n]]与[h_n]， h_j表示了agent j与agent i的关系
        # 输入的iputs维度为(n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)，
        # 即对于batch_size条数据，输入每个agent与其他n_agents - 1个agents的hidden_state的连接
        self.hard_bi_GRU = nn.GRU(args.rnn_hidden_dim * 2, args.rnn_hidden_dim, bidirectional=True)
        # 对h_j进行分析，得到agent j对于agent i的权重，输出两维，经过gumble_softmax后取其中一维即可，如果是0则不考虑agent j，如果是1则考虑
        self.hard_encoding = nn.Linear(args.rnn_hidden_dim * 2, 2)  # 乘2因为是双向GRU，hidden_state维度为2 * hidden_dim

        # Soft
        self.q = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.k = nn.Linear(args.rnn_hidden_dim, args.attention_dim, bias=False)
        self.v = nn.Linear(args.rnn_hidden_dim, args.attention_dim)

        # Decoding 输入自己的h_i与x_i，输出自己动作的概率分布
        self.decoding = nn.Linear(args.rnn_hidden_dim + args.attention_dim, args.n_actions)
        self.args = args
        self.input_shape = input_shape

    def forward(self, obs, hidden_state):
        size = obs.shape[0]  # batch_size * n_agents
        # 先对obs编码
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)

        # 经过自己的GRU得到h
        h_out = self.h(obs_encoding, h_in)  # (batch_size * n_agents, args.rnn_hidden_dim)

        # Hard Attention，GRU和GRUCell不同，输入的维度是(序列长度,batch_size, dim)
        if self.args.hard:
            # Hard Attention前的准备
            h = h_out.reshape(-1, self.args.n_agents, self.args.rnn_hidden_dim)  # 把h转化出n_agents维度，(batch_size, n_agents, rnn_hidden_dim)
            input_hard = []
            for i in range(self.args.n_agents):
                h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
                h_hard_i = []
                for j in range(self.args.n_agents):  # 对于agent i，把自己的h_i与其他agent的h分别拼接
                    if j != i:
                        h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
                # j 循环结束之后，h_hard_i是一个list里面装着n_agents - 1个维度为(batch_size, rnn_hidden_dim * 2)的tensor
                h_hard_i = torch.stack(h_hard_i, dim=0)
                input_hard.append(h_hard_i)
            # i循环结束之后，input_hard是一个list里面装着n_agents个维度为(n_agents - 1, batch_size, rnn_hidden_dim * 2)的tensor
            input_hard = torch.stack(input_hard, dim=-2)
            # 最终得到维度(n_agents - 1, batch_size * n_agents, rnn_hidden_dim * 2)，可以输入了
            input_hard = input_hard.view(self.args.n_agents - 1, -1, self.args.rnn_hidden_dim * 2)

            h_hard = torch.zeros((2 * 1, size, self.args.rnn_hidden_dim))  # 因为是双向GRU，每个GRU只有一层，所以第一维是2 * 1
            if self.args.cuda:
                h_hard = h_hard.cuda()
            h_hard, _ = self.hard_bi_GRU(input_hard, h_hard)  # (n_agents - 1,batch_size * n_agents,rnn_hidden_dim * 2)
            h_hard = h_hard.permute(1, 0, 2)  # (batch_size * n_agents, n_agents - 1, rnn_hidden_dim * 2)
            h_hard = h_hard.reshape(-1, self.args.rnn_hidden_dim * 2)  # (batch_size * n_agents * (n_agents - 1), rnn_hidden_dim * 2)

            # 得到hard权重, (n_agents, batch_size, 1,  n_agents - 1)，多出一个维度，下面加权求和的时候要用
            hard_weights = self.hard_encoding(h_hard)
            hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
            # print(hard_weights)
            hard_weights = hard_weights[:, 1].view(-1, self.args.n_agents, 1, self.args.n_agents - 1)
            hard_weights = hard_weights.permute(1, 0, 2, 3)

        else:
            hard_weights = torch.ones((self.args.n_agents, size // self.args.n_agents, 1, self.args.n_agents - 1))
            if self.args.cuda:
                hard_weights = hard_weights.cuda()

        # Soft Attention
        q = self.q(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        v = f.relu(self.v(h_out)).reshape(-1, self.args.n_agents, self.args.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        x = []
        for i in range(self.args.n_agents):
            q_i = q[:, i].view(-1, 1, self.args.attention_dim)  # agent i的q，(batch_size, 1, args.attention_dim)
            k_i = [k[:, j] for j in range(self.args.n_agents) if j != i]  # 对于agent i来说，其他agent的k
            v_i = [v[:, j] for j in range(self.args.n_agents) if j != i]  # 对于agent i来说，其他agent的v

            k_i = torch.stack(k_i, dim=0)  # (n_agents - 1, batch_size, args.attention_dim)
            k_i = k_i.permute(1, 2, 0)  # 交换三个维度，变成(batch_size, args.attention_dim， n_agents - 1)
            v_i = torch.stack(v_i, dim=0)
            v_i = v_i.permute(1, 2, 0)

            # (batch_size, 1, attention_dim) * (batch_size, attention_dim，n_agents - 1) = (batch_size, 1，n_agents - 1)
            score = torch.matmul(q_i, k_i)

            # 归一化
            scaled_score = score / np.sqrt(self.args.attention_dim)

            # softmax得到权重
            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)

            # 加权求和，注意三个矩阵的最后一维是n_agents - 1维度，得到(batch_size, args.attention_dim)
            x_i = (v_i * soft_weight * hard_weights[i]).sum(dim=-1)
            x.append(x_i)

        # 合并每个agent的h与x
        x = torch.stack(x, dim=1).reshape(-1, self.args.attention_dim)  # (batch_size * n_agents, args.attention_dim)
        final_input = torch.cat([h_out, x], dim=-1)
        output = self.decoding(final_input)

        return output, h_out

