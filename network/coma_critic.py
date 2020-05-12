import torch.nn as nn
import torch.nn.functional as F

'''
输入当前的状态、当前agent的obs、其他agent执行的动作、当前agent的编号对应的one-hot向量、所有agent上一个timestep执行的动作
输出当前agent的所有可执行动作对应的联合Q值——一个n_actions维向量
'''


class ComaCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(ComaCritic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, self.args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q