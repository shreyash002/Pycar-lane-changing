import torch.nn as nn
import torch.nn.functional as F
import torch
import random
from collections import namedtuple

# custom weights initialization
def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)

class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self):

        self.memory_capacity = 40000
        self.capacity = self.memory_capacity
        self.memory = []
        self.position = 0

    def length(self):
        return len(self.memory)

    def push_transition(self, *args):

        if self.length() < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # for the cyclic buffer

    def sample_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_channels = 12
        self.conv_filters = [16, 32, 64, 64] 
        self.num_classes = 5

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=self.conv_filters[0], kernel_size=7, stride=2, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(self.conv_filters[0])

        self.conv2 = nn.Conv2d(in_channels=self.conv_filters[0], out_channels=self.conv_filters[1], kernel_size=5, stride=2, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(self.conv_filters[1])

        self.conv3 = nn.Conv2d(in_channels=self.conv_filters[1], out_channels=self.conv_filters[2], kernel_size=3, stride=2, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(self.conv_filters[2])

        self.conv4 = nn.Conv2d(in_channels=self.conv_filters[2], out_channels=self.conv_filters[3], kernel_size=3, stride=1, padding=0, bias=True)
        self.bn4 = nn.BatchNorm2d(self.conv_filters[3])

        self.linear = nn.Linear(5632, 512)

        self.linear2 = nn.Linear(512, self.num_classes)

        self.apply(weights_init)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.linear(x.view(x.size(0), -1))
        out = self.linear2(x)

        return out