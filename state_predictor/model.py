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


class SPNMultihead(nn.Module):
    def __init__(self, Nin):
        super(SPNMultihead, self).__init__()

        # Shared trunk
        self.backend = nn.Sequential(
            nn.Linear(Nin, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Head for spars
        self.head_spars = nn.Sequential(
            nn.Linear(256, 1)
        )

        # Head for dmap
        self.head_dmap = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        shared_out = self.backend(x)

        out_spars = self.head_spars(shared_out)
        out_dmap = self.head_dmap(shared_out)

        # return shape: (batch, 2)
        return torch.cat([out_spars, out_dmap], dim=1)

