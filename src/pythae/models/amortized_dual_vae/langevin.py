"""
Persistent Contrastive Divergence with Langevin dynamics.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .monomials import MonomialBasis


class LangevinPCD:
    """Short-run persistent Langevin sampler.

    The sampler keeps a persistent buffer per mini-batch entry and performs a fixed number of
    Langevin steps before returning samples.
    """

    def __init__(
        self,
        latent_dim: int,
        steps: int = 30,
        step_size: float = 1e-2,
        *,
        n_samples: int = 1,
        c: float = 0.5,
        reinit_prob: float = 0.1,
        noise_scale: Optional[float] = None,
        update_clamp: float = 3.0,
    ):
        if steps <= 0:
            raise ValueError("`steps` must be strictly positive.")

        if step_size <= 0:
            raise ValueError("`step_size` must be strictly positive.")

        self.latent_dim = latent_dim
        self.steps = steps
        self.step_size = step_size
        self.n_samples = n_samples
        self.c = c
        self.reinit_prob = reinit_prob
        self.noise_scale = math.sqrt(2.0 * step_size) if noise_scale is None else noise_scale
        self.update_clamp = update_clamp

        self._buffer: Optional[torch.Tensor] = None

    def reset(self):
        """Forget the persistent buffer."""
        self._buffer = None

    def update_buffer(self, new_buffer: torch.Tensor):
        """Manually set the persistent buffer."""
        self._buffer = new_buffer.detach()

    def sample(self, lam: torch.Tensor, basis: MonomialBasis) -> torch.Tensor:
        """Draw samples using persistent Langevin dynamics.

        Args:
            lam (torch.Tensor): Natural parameters of shape ``[B, K]``.
            basis (MonomialBasis): Monomial basis used within the energy function.

        Returns:
            torch.Tensor: Samples with shape ``[B, n_samples, latent_dim]``.
        """
        if lam.dim() != 2:
            raise ValueError("`lam` must be of shape [batch, K].")

        batch_size, _ = lam.shape
        device = lam.device
        dtype = lam.dtype

        with torch.no_grad():
            self._ensure_buffer(batch_size, device, dtype)

            samples = []
            buffer = self._buffer

            for _ in range(self.n_samples):
                buffer = self._run_chain(buffer, lam.detach(), basis)
                samples.append(buffer.unsqueeze(1))

            self._buffer = buffer.detach()

            return torch.cat(samples, dim=1)

    def _run_chain(
        self, z: torch.Tensor, lam: torch.Tensor, basis: MonomialBasis
    ) -> torch.Tensor:
        current = z

        for _ in range(self.steps):
            current = self._langevin_step(current, lam, basis)

        return current

    def _langevin_step(
        self, z: torch.Tensor, lam: torch.Tensor, basis: MonomialBasis
    ) -> torch.Tensor:
        with torch.enable_grad():
            z = z.clone().detach().requires_grad_(True)
            energy = self._energy(z, lam, basis)
            grad_z = torch.autograd.grad(energy.sum(), z, create_graph=False)[0]

        noise = torch.randn_like(z) * self.noise_scale
        updated = z.detach() - self.step_size * grad_z.detach() + noise
        updated.clamp_(- self.update_clamp, self.update_clamp)
        return updated

    def _energy(self, z: torch.Tensor, lam: torch.Tensor, basis: MonomialBasis) -> torch.Tensor:
        quad = 0.5 * self.c * (z ** 2).sum(dim=-1)
        features = basis(z)
        return quad - (lam * features).sum(dim=-1)

    def _ensure_buffer(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if (
            self._buffer is None
            or self._buffer.shape[0] != batch_size
            or self._buffer.device != device
            or self._buffer.dtype != dtype
        ):
            self._buffer = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)

        elif self.reinit_prob > 0:
            mask = torch.rand(batch_size, device=device) < self.reinit_prob

            if mask.any():
                num_reset = mask.sum()
                self._buffer[mask] = torch.randn(
                    num_reset, self.latent_dim, device=device, dtype=dtype
                )
