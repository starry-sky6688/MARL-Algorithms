import torch.nn as nn
import torch.nn.functional as F
import torch


class DyAN(nn.Module):
    # obs_shape应该是obs_shape+n_actions+n_agents，还要输入当前agent的上一个动作和agent编号，这样就可以只使用一个神经网络
    def __init__(self, input_shape, args):
        super(DyAN, self).__init__()
        self.args = args
        hidden_dim = 32
        self.movement_feature_shape = 4
        self.enemy_shape = 5
        self.ally_shape = 5
        self.unit_shape = 1
        if self.args.map == '3m':
            self.enemy_num = 3
            self.ally_num = 2
        if self.args.map == '8m':
            self.enemy_num = 8
            self.ally_num = 7

        # 解码对自己的obs
        self.fc_self = nn.Linear(self.movement_feature_shape + self.unit_shape + args.n_actions, hidden_dim)

        # 解码对敌人的obs
        self.fc_enemy = nn.Linear(self.enemy_shape, hidden_dim)

        # 解码对队友的obs
        self.fc_ally = nn.Linear(self.ally_shape, hidden_dim)

        # 解码自己、敌人、队友的联合信息
        self.fc_final = nn.Linear(hidden_dim * 3, args.rnn_hidden_dim)

        # 解码最终向量
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.output = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc_final.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        obs_self = torch.cat([obs[:, :4], obs[:, -(self.unit_shape + self.args.n_actions + self.args.n_agents): -self.args.n_agents]], dim=-1)
        obs_enemy = [obs[:, 4 + i * 5: 4 + (i + 1) * 5] for i in range(self.enemy_num)]
        obs_ally = [obs[:, 4 + 5 * self.enemy_num + i * 5: 4 + 5 * self.enemy_num + (i + 1) * 5] for i in range(self.ally_num)]

        # TODO 输入到神经网络
        x_self = F.relu(self.fc_self(obs_self))
        x_enemy = [F.relu(self.fc_enemy(obs_enemy[i])) for i in range(self.enemy_num)]
        x_ally = [F.relu(self.fc_ally(obs_ally[i])) for i in range(self.ally_num)]

        x_enemy_sum, x_ally_sum = torch.zeros_like(x_enemy[0]), torch.zeros_like(x_ally[0])
        for i in range(self.enemy_num):
            x_enemy_sum += x_enemy[i]
        for i in range(self.ally_num):
            x_ally_sum += x_ally[i]

        x_final = torch.cat([x_self, x_enemy_sum, x_ally_sum], dim=-1)
        x_final = F.relu(self.fc_final(x_final))

        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x_final, h_in)
        q = self.output(h)
        return q, h
