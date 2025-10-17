from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..base.base_utils import ModelOutput
from ..nn import BaseEncoder


class BaseLambdaNet(nn.Module):
    """Base class for networks producing natural parameters."""

    def __init__(self):
        super().__init__()

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MomentsEncoderMLP(BaseEncoder):
    """Default encoder producing target moment vectors."""

    def __init__(self, config):
        super().__init__()

        if config.input_dim is None:
            raise AttributeError(
                "No input dimension provided! `input_dim` must be set in the config."
            )

        self.input_dim = config.input_dim
        self.output_dim = config.moment_dim
        self.output_scale = config.encoder_output_scale

        layers = nn.ModuleList()
        in_dim = int(np.prod(config.input_dim))

        for _ in range(config.encoder_num_layers):
            layers.append(nn.Sequential(nn.Linear(in_dim, config.encoder_hidden_dim), nn.ReLU()))
            in_dim = config.encoder_hidden_dim

        self.layers = layers
        self.depth = len(layers)
        self.head = nn.Linear(in_dim, self.output_dim)

    def forward(self, x: torch.Tensor, output_layer_levels=None) -> ModelOutput:
        output = ModelOutput()
        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= level > 0 or level == -1 for level in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 not in output_layer_levels:
                max_depth = max(output_layer_levels)

        out = x.reshape(x.shape[0], -1)

        for idx in range(max_depth):
            out = self.layers[idx](out)

            if output_layer_levels is not None and (idx + 1) in output_layer_levels:
                output[f"embedding_layer_{idx+1}"] = out

        moments = torch.tanh(self.head(out)) * self.output_scale
        output["embedding"] = moments

        return output


class LambdaNetMLP(BaseLambdaNet):
    """Default natural-parameter generator."""

    def __init__(self, config):
        super().__init__()

        self.input_dim = config.moment_dim
        self.output_dim = config.moment_dim

        activation = nn.ReLU if config.lambda_activation == "relu" else nn.Tanh

        layers = []
        in_dim = self.input_dim

        for _ in range(config.lambda_num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, config.lambda_hidden_dim),
                    activation(),
                ]
            )
            in_dim = config.lambda_hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        return self.net(moments)

