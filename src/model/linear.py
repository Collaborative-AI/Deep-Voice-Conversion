import torch
import torch.nn as nn
import math
from .model import init_param, make_loss


class Linear(nn.Module):
    def __init__(self, data_size, target_size):
        super().__init__()
        input_size = math.prod(data_size)
        self.linear = nn.Linear(input_size, target_size)

    def feature(self, x):
        x = x.reshape(x.size(0), -1)
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


def linear(cfg):
    data_size = cfg['data_size']
    target_size = cfg['target_size']
    model = Linear(data_size, target_size)
    model.apply(init_param)
    return model