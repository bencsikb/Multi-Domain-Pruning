import torch
from torch import nn
from torch.distributions import Categorical

#TODO return types

class criticNet(nn.Module):
    def __init__(self, Nin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Nin, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        v = self.net(x)
        return v.squeeze(-1)   # [B]
        

class actorNet(nn.Module):
    def __init__(self, Nin, n_actions=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Nin, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.logits = nn.Linear(256, n_actions)

    def forward(self, x):
        h = self.net(x)
        logits = self.logits(h)
        dist = Categorical(logits=logits) 
        return dist, logits
