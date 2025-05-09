from config import cfg
import numpy as np


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']

    # cfg['batch_size'] = 1
    cfg['batch_size'] = 250
    cfg['step_period'] = 1
    cfg['num_steps'] = 80000
    cfg['eval_period'] = 200
    # cfg['num_epochs'] = 400
    cfg['collate_mode'] = 'dict'

    cfg['sample_rate'] = 16000
    # cfg['segment_seconds'] = 1
    cfg['segment_length'] = 38100

    # cfg['wav_length'] = cfg['sample_rate'] * cfg['segment_seconds']

    # mel_shape = [1, cfg['n_mels'], int(cfg['wav_length'] / cfg['hop_length']) + 1]

    cfg['model'] = {}
    cfg['model']['sample_rate'] = cfg['sample_rate']
    # cfg['model']['segment_seconds'] = cfg['segment_seconds']
    cfg['model']['model_name'] = cfg['model_name']
    data_size = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                 'CIFAR100': [3, 32, 32], 'VCTK': [-1]}
    target_size = {'MNIST': 10, 'FashionMNIST': 10, 'SVHN': 10,
                   'CIFAR10': 10, 'CIFAR100': 100, 'VCTK': 1}
    cfg['model']['data_size'] = data_size[cfg['data_name']]
    cfg['model']['target_size'] = target_size[cfg['data_name']]
    cfg['model']['linear'] = {}
    cfg['model']['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['model']['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet10'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['model']['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['model']['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}

    cfg['model']['mi'] = {'club': {'hidden_size': 64}, 'mine': {'hidden_size': 64}, 'num_steps': 5}
    cfg['model']['mainvc'] = {
        "SpeakerEncoder": {
            "c_in": 80,
            "c_h": 64,
            "c_out": 64,
            "c_bank": 64,
            "kernel_size": 5,
            "n_conv_blocks": 6,
            "n_dense_blocks": 6,
            "subsample": [1, 2, 1, 2, 1, 2],
            "act": "relu",
            "dropout_rate": 0
        },
        "ContentEncoder": {
            "c_in": 80,
            "c_h": 64,
            "c_out": 64,
            "c_bank": 64,
            "kernel_size": 5,
            "n_conv_blocks": 6,
            "subsample": [1, 2, 1, 2, 1, 2],
            "act": "relu",
            "dropout_rate": 0
        },
        "Decoder": {
            "c_in": 64,
            "c_cond": 64,
            "c_h": 64,
            "c_out": 80,
            "kernel_size": 5,
            "n_conv_blocks": 6,
            "upsample": [2, 1, 2, 1, 2, 1],
            "act": "relu",
            "sn": False,
            "dropout_rate": 0
        },
        # "CMI": {
        #     "mine": 64,
        #     "club": 64
        # },
        # "lambda": {
        #     "lambda_rec": 10,
        #     "lambda_kl": 1,
        #     "lambda_sia": 5,
        #     "lambda_mi": 1
        # },
        'mel': {
            'preemph': 0.97,
            'n_fft': [2048],
            'win_length': [1200],
            'hop_length': [300],
            'n_mels': [80],
            'f_min': 80,
            'f_max': 8000,
            'shuffle_size': 20
        }
    }
    cfg['model']['regularization'] = {'rec': 10, 'kl': 1, 'sia': 5, 'mi': 1}

    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 1e-4
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'test_in': cfg['batch_size'],
                                           'test_out': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'CosineAnnealingLR'
    return
