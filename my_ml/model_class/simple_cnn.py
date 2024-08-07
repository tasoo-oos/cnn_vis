import torch
import torch.nn as nn
from typing import List, Tuple

class CNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            cnn_layers: List[Tuple[int, int]],
            cnn_dropout_rate: float = 0.3
    ):
        super(CNN, self).__init__()
        self.features = self._make_feature_layers(in_channels, cnn_layers, cnn_dropout_rate)
        self.classifier = self._make_classifier(cnn_layers[-1][0], num_classes)

    def _make_feature_layers(self, in_channels: int, cnn_layers: List[Tuple[int, int]], dropout_rate: float) -> nn.Sequential:
        layers = []
        for cnn_stack in cnn_layers:
            for _ in range(cnn_stack[1]):
                layers.extend(self._make_conv_block(in_channels, cnn_stack[0], dropout_rate))
                in_channels = cnn_stack[0]
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        return nn.Sequential(*layers)

    @staticmethod
    def _make_conv_block(in_channels: int, out_channels: int, dropout_rate: float) -> List[nn.Module]:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate)
        ]

    @staticmethod
    def _make_classifier(in_features: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
