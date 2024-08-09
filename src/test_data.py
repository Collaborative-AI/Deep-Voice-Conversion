import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, Stats, makedir_exist_ok, process_control
import soundfile as sf
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np

# sf.read('../data/VCTK/raw/wav8_silence_trimmed/p286/p286_001.wav')
# data_names = ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'VCTK'] # ADD VCTK
data_name = 'VCTK' # ADD VCTK

# cfg['seed'] = 0
# cfg['tag'] = 'make_dataset'
# process_control()

dataset = make_dataset(data_name)
# print(dataset['val'])
# process_dataset(dataset)
data_loader = make_data_loader(dataset, {'train':16, 'test':16, 'val':16}, shuffle=False)
# k=0
# print(max([item['wave'].size(1) for item in data_loader['train']]))
# count = 0
# sse = 0
# mean = 34167.14074396903 - 10000
# maximum = 136464 - 10000
for item in tqdm(data_loader['train']):
    # print(item['wave'].size())
    # print(item.keys())
    # print(np.array(item['data'])[0].shape)
    # plt.plot(np.arange(item['data'].size()[1]), np.array(item['data'])[0])
    # plt.show()
    pass

print("done")
# variance = sse/count
# print(f"variance: {variance}")
# print(f"stdev: {math.sqrt(variance)}")