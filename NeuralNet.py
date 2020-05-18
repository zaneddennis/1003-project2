import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(in_features=5000, out_features=1000)
        self.l2 = nn.Linear(in_features=1000, out_features=100)
        self.l3 = nn.Linear(in_features=100, out_features=20)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
        
