# config.py
from torch import nn

DATASET_CONFIGS = {
    'MNIST': {
        'in_channels': 1,
        'num_classes': 10,
    },
    'FashionMNIST': {
        'in_channels': 1,
        'num_classes': 10,
    },
    'CIFAR10': {
        'in_channels': 3,
        'num_classes': 10,
    }
}

MODEL_CONFIGS = {
    'CNN': {
        'cnn_layers': [(8, 3), (16, 3), (32, 3)]
    },
    'DenseNet': {
        'block_config': (4, 8, 16),
        'growth_rate': 16,
        'num_init_features': 64,
        'bn_size': 2,
        'drop_rate': 0.4,
    },
    'ResNet': {
        'model_config': ((16, 3), (32, 3), (64, 3), (128, 3)),
        'activation': nn.GELU
    }
}

DATASET_TO_MODEL = {
    'MNIST': 'ResNet',
    'FashionMNIST': 'ResNet',
    'CIFAR10': 'ResNet'
}
