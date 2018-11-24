import torch
import torch.nn as nn
import torch.nn.functional as F


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 23, 5)
        #self.fc1 = nn.Linear(23 * 5 * 5, 120)
        self.fc1 = nn.Linear(28 * 28 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 23 * 5 * 5)
        x = x.view(-1, 28*28*1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
