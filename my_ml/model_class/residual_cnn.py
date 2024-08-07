import torch
import torch.nn as nn
from typing import List, Tuple, Callable

class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            use_dropout: bool = False,
            dropout_rate: float = 0,
            activation: Callable = nn.GELU):

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        self.down_sample = None
        if in_channels != out_channels or stride != 1:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            use_dropout: bool = False,
            use_first_layer: bool = False,
            initial_channels: int = 8,
            dropout_rate: float = 0,
            model_config: Tuple[Tuple[int, int], ...] = ((32, 2), (16, 2), (8, 2)),
            activation: Callable = nn.ReLU):
        """
        ResNet model implementation.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            use_dropout (bool): Whether to use dropout.
            dropout_rate (float): Dropout rate.
            model_config (Tuple[Tuple[int, int], ...]): Configuration for each layer (channels, blocks).
            activation (Callable): Activation function to use.
        """
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        if use_first_layer:
            self.layers = self._make_first_layer(initial_channels)
        else:
            self.layers = nn.Identity()
        self.model_config = model_config
        self.activation = activation

        self.layers = self._make_layers()
        self._initialize_weights()

    def _make_mid_layers(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = nn.ModuleList()
        layers.append(ResidualBlock(in_channels, out_channels, stride=2, use_dropout=self.use_dropout,
                                    dropout_rate=self.dropout_rate, activation=self.activation))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_dropout=self.use_dropout,
                                        dropout_rate=self.dropout_rate, activation=self.activation))
        return nn.Sequential(*layers)

    def _make_last_layers(self, in_channels: int):
        layers = nn.ModuleList()

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, self.num_classes))

        return nn.Sequential(*layers)

    def _make_layers(self):
        layers = nn.ModuleList()

        in_channels = self.in_channels
        for out_channels, num_blocks in self.model_config:
            layers.append(self._make_mid_layers(in_channels, out_channels, num_blocks))
            in_channels = out_channels

        layers.append(self._make_last_layers(in_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)