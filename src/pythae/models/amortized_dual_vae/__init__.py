"""Amortized dual-solver VAE with energy-based latent distribution."""

from .amortized_dual_vae_config import AmortizedDualVAEConfig
from .amortized_dual_vae_model import AmortizedDualVAE

__all__ = ["AmortizedDualVAE", "AmortizedDualVAEConfig"]

