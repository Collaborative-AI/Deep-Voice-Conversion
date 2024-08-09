import dataset
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg
from torch.nn.utils.rnn import pad_sequence

data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))}


def make_dataset(data_name, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['VCTK']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train")'.format(data_name)) 
        dataset_['test'] = eval('dataset.{}(root=root, split="test")'.format(data_name))
        dataset_['val'] = eval('dataset.{}(root=root, split="test")'.format(data_name))
    else: 
        raise ValueError('Not valid dataset name')
    if verbose and data_name not in ['VCTK']:
        dataset_['train'] = eval('dataset.{}(root=root, split="train", '
                                 'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name)) 
        dataset_['test'] = eval('dataset.{}(root=root, split="test", '
                                'transform=dataset.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset_['train'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset_['test'].transform = dataset.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        print('data ready')
    return dataset_


def input_collate(input):
    first = input[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in input])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in input]))
            else:
                batch[k] = torch.tensor([f[k] for f in input])
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                a = dataset[k]
                b = batch_size[k]
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset):
    processed_dataset = dataset
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    return processed_dataset


# class Collater(object):
#     """
#     Args:
#       adaptive_batch_size (bool): if true, decrease batch size when long data comes.
#     """

#     def __init__(self):
#         self.text_pad_index = 0
#         self.min_mel_length = 192
#         self.max_mel_length = 192
        

#     def __call__(self, batch):
#         # batch[0] = wave, mel, text, f0, speakerid
#         batch_size = len(batch)

#         # sort by mel length
#         lengths = [b[1].shape[1] for b in batch]
#         batch_indexes = np.argsort(lengths)[::-1]
#         batch = [batch[bid] for bid in batch_indexes]

#         nmels = batch[0][1].size(0)
#         max_mel_length = max([b[1].shape[1] for b in batch])
#         max_text_length = max([b[2].shape[0] for b in batch])
        
#         labels = torch.zeros((batch_size)).long()
#         mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
#         texts = torch.zeros((batch_size, max_text_length)).long()
#         input_lengths = torch.zeros(batch_size).long()
#         output_lengths = torch.zeros(batch_size).long()
#         ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
#         ref_labels = torch.zeros((batch_size)).long()
#         paths = ['' for _ in range(batch_size)]
        
#         for bid, (wave, label, mel, text, ref_mel, ref_label, path) in enumerate(batch):
#             mel_size = mel.size(1)
#             text_size = text.size(0)
#             labels[bid] = label
#             mels[bid, :, :mel_size] = mel
#             texts[bid, :text_size] = text
#             input_lengths[bid] = text_size
#             output_lengths[bid] = mel_size
#             paths[bid] = path
            
#             ref_mel_size = ref_mel.size(1)
#             ref_mels[bid, :, :ref_mel_size] = ref_mel
            
#             ref_labels[bid] = ref_label
                        
#         return wave, texts, input_lengths, mels, output_lengths, labels, ref_mels, ref_labels