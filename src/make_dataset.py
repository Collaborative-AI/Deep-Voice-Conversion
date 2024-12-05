import os
import torch
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset
from module import process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['VCTK']
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    with torch.no_grad():
        for data_name in data_names:
            dataset = make_dataset(data_name)
            process_dataset(dataset)
            cfg['step'] = 0
            data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
            print('num samples: {}'.format(len(dataset['train'])), 'num batches: {}'.format(len(data_loader['train'])))
            for i, input in enumerate(data_loader['train']):
                print('audio shape: {}'.format(input['audio'].shape))
                break
