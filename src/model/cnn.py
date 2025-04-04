import torch
import torch.nn as nn
from .model import init_param, make_loss


class CNN(nn.Module):
    def __init__(self, data_size, hidden_size, target_size):
        super().__init__()
        blocks = [nn.Conv2d(data_size[0], hidden_size[0], 3, 1, 1),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten()])
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(hidden_size[-1], target_size)

    def feature(self, x):
        x = self.blocks(x)
        return x

    def output(self, x):
        x = self.linear(x)
        return x

    def f(self, x):
        x = self.feature(x)
        x = self.output(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = self.f(x)
        output['target'] = x
        output['loss'] = make_loss(output, input)
        return output


def cnn(cfg):
    data_size = cfg['data_size']
    target_size = cfg['target_size']
    hidden_size = cfg['cnn']['hidden_size']
    model = CNN(data_size, hidden_size, target_size)
    model.apply(init_param)
    return model
