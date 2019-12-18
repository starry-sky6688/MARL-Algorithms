import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
因为这里使用的是RNN，每次还需要上一次的hidden_state，对于一个episode的数据，每个obs要选择动作都需要上一次的hidden_state
所以就不能直接随机抽取一批经验输入到神经网络，因此这里需要用一批episode，每次都传入这一批episode的同一个位置的transition，
这样的话就可以将hidden_state保存下来，然后下一次传入的就是下一条经验
'''


class RNN(nn.Module):
    # obs_shape应该是obs_shape+n_actions+n_agents，还要输入当前agent的上一个动作和agent编号，这样就可以只使用一个神经网络
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
