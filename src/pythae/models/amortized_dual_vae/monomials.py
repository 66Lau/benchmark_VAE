"""
Utilities for building polynomial feature bases over latent variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Tuple

import torch


MultiIndex = Tuple[int, ...]


@lru_cache(maxsize=None)
def enumerate_multi_indices(
    latent_dim: int, order: int, exclude_constant: bool = True
) -> Tuple[MultiIndex, ...]:
    """Enumerate the multi-indices for a polynomial basis.

    Args:
        latent_dim (int): Dimension of the latent variable ``z``.
        order (int): Maximum total degree to include.
        exclude_constant (bool): If ``True`` the all-zeros index is removed.

    Returns:
        Tuple[MultiIndex, ...]: Multi-indices sorted by increasing total degree then lexicographic
            order.
    """
    if latent_dim <= 0:
        raise ValueError("`latent_dim` must be strictly positive.")

    if order <= 0:
        raise ValueError("`order` must be strictly positive.")

    indices: List[MultiIndex] = []

    for total_degree in range(0, order + 1):
        for alpha in _enumerate_degree(latent_dim, total_degree):
            indices.append(alpha)

    if exclude_constant and indices and all(v == 0 for v in indices[0]):
        indices = indices[1:]

    return tuple(indices)


def _enumerate_degree(latent_dim: int, degree: int) -> Iterable[MultiIndex]:
    if latent_dim == 1:
        yield (degree,)
        return

    for value in range(degree + 1):
        for rest in _enumerate_degree(latent_dim - 1, degree - value):
            yield (value,) + rest


class MonomialBasis:
    """Vectorised evaluation of a monomial basis ``T(z)``.

    The class precomputes the exponent patterns so that repeated calls are cheap.
    """

    def __init__(self, latent_dim: int, order: int, exclude_constant: bool = True):
        self.latent_dim = latent_dim
        self.order = order
        self.exclude_constant = exclude_constant

        self._multi_indices: Tuple[MultiIndex, ...] = enumerate_multi_indices(
            latent_dim, order, exclude_constant=exclude_constant
        )
        self._alpha_tensor_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

        if len(self._multi_indices) == 0:
            raise ValueError(
                "Polynomial basis is empty. Ensure `order` > 0 or allow constant term."
            )

    @property
    def num_features(self) -> int:
        return len(self._multi_indices)

    def _alpha_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        tensor = self._alpha_tensor_cache.get(key)

        if tensor is None:
            tensor = torch.tensor(
                self._multi_indices, device=device, dtype=dtype, requires_grad=False
            )
            self._alpha_tensor_cache[key] = tensor

        return tensor

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the basis for a batch of latent vectors.

        Args:
            z (torch.Tensor): Tensor with final dimension ``latent_dim``. Additional leading
                dimensions are preserved in the output.

        Returns:
            torch.Tensor: Tensor of shape ``(*z.shape[:-1], K)`` with ``K`` monomial features.
        """
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected last dimension {self.latent_dim}, got {z.shape[-1]}."
            )

        orig_shape = z.shape[:-1]
        flat_z = z.reshape(-1, self.latent_dim)
        alpha = self._alpha_tensor(z.device, z.dtype)

        z_expanded = flat_z.unsqueeze(1)  # [N, 1, latent_dim]
        monomials = torch.pow(z_expanded, alpha.unsqueeze(0)).prod(dim=-1)

        return monomials.reshape(*orig_shape, self.num_features)

    def __len__(self) -> int:
        return self.num_features
