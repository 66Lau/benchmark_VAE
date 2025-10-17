from dataclasses import field
from math import comb
from typing import Optional

from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from ..base.base_config import BaseAEConfig


@dataclass
class AmortizedDualVAEConfig(BaseAEConfig):
    """Configuration for the Amortized Dual Solver VAE."""

    reconstruction_loss: Literal["bce", "mse"] = "mse"
    polynomial_order: int = 2
    energy_scale: float = 0.1

    langevin_steps: int = 30
    langevin_step_size: float = 1e-3
    langevin_n_samples: int = 4
    langevin_reinit_prob: float = 0.1
    langevin_noise_scale: Optional[float] = None
    langevin_update_clamp: float = 3.0

    score_weight: float = 0.1
    dual_weight: float = 0.001
    moment_weight: float = 1.0
    lambda_reg_weight: float = 1
    moment_reg_weight: float = 0.0

    encoder_hidden_dim: int = 512
    encoder_num_layers: int = 2
    encoder_output_scale: float = 5.0

    lambda_hidden_dim: int = 512
    lambda_num_layers: int = 2
    lambda_activation: Literal["relu", "tanh"] = "relu"

    uses_default_lambda_net: bool = True
    moment_dim: int = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.latent_dim <= 0:
            raise ValueError("`latent_dim` must be a strictly positive integer.")

        if self.polynomial_order <= 0:
            raise ValueError("`polynomial_order` must be a strictly positive integer.")

        self.moment_dim = comb(self.latent_dim + self.polynomial_order, self.polynomial_order) - 1

        if self.moment_dim <= 0:
            raise ValueError("Computed moment dimension must be positive. Check config values.")
