import torch
from torch import nn
from torch.distributions import Categorical

#TODO return types

def init_weights(m):
    print("inxavier")
    if isinstance(m, nn.Linear):
        print("Xavier")
        torch.nn.init.xavier_uniform(m.weight)

class criticNet(nn.Module):
    def __init__(self, Nin, Nout):
        super(criticNet, self).__init__()

        self.L1 = nn.Linear(Nin, 256)
        self.L2 = nn.Linear(256,512)
        self.L3 = nn.Linear(512, Nout)

    def forward(self, x):

        x1 = torch.relu(self.L1(x))
        x2 = torch.relu(self.L2(x1))
        x3 = self.L3(x2)

        return x3


class actorNet(nn.Module):
    def __init__(self, Nin, Nout):
        super(actorNet, self).__init__()

        self.L1 = nn.Linear(Nin, 512)
        self.L2 = nn.Linear(512, 1024)
        self.L3 = nn.Linear(1024, 256)
        self.L4 = nn.Linear(256, Nout)

    def forward(self, x):
        #print(x)
        x1 = torch.sigmoid(self.L1(x))
        x2 = torch.sigmoid(self.L2(x1))
        x3 = torch.sigmoid(self.L3(x2))
        probs = torch.softmax(self.L4(x3), dim=1)
        log_softmax = torch.log_softmax(self.L4(x3), dim=1)

        distribution  = Categorical(probs)

        return probs, distribution, log_softmax