import torch
from torch import nn

class SPN(nn.Module):
    def __init__(self, Nin, Nout):
        super(SPN, self).__init__()

        self.L1 = nn.Linear(Nin, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.L2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.L3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.L4 = nn.Linear(256, Nout)

    def forward(self, x):

        x1 = torch.relu(self.bn1(self.L1(x)))
        x2 = torch.relu(self.bn2(self.L2(x1)))
        x3 = torch.relu(self.bn3(self.L3(x2)))
        x4 = self.L4(x3)

        return x4