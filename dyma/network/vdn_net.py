import torch.nn as nn
import torch


class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)

