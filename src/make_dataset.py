import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    # data_names = ['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'VCTKTime', 'VCTKMel']
    # data_names = ['VCTKTime', 'VCTKMel']
    data_names = ['VCTKTime']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    with torch.no_grad():
        for data_name in data_names:
            dataset = make_dataset(data_name)
            process_dataset(dataset)
            cfg['step'] = 0
            data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
            stats = Stats(dim=dim)
            print(len(dataset['train']))
            for i, input in enumerate(data_loader['train']):
                print(i, input['data'].shape)
                stats.update(input['data'])
            print(data_name, stats)
            makedir_exist_ok(stats_path)
            save(stats, os.path.join(stats_path, '{}'.format(data_name)), 'torch')
