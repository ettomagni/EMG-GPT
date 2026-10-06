"""Frozen RVQ lookup used by the frame encoder."""

import torch
from torch.nn import functional as F


def dequantize_tokens(codes: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Sum the K frozen RVQ vectors for each channel/branch."""
    n_branches, n_scales, _, code_dim = codebooks.shape
    if codes.shape[-2] != n_branches or codes.shape[-1] != n_scales:
        raise ValueError(
            f"codes branch/scale ({codes.shape[-2]},{codes.shape[-1]}) != codebooks ({n_branches},{n_scales})"
        )
    out = torch.zeros((*codes.shape[:-1], code_dim), dtype=codebooks.dtype, device=codes.device)
    for branch in range(n_branches):
        for scale in range(n_scales):
            out[..., branch, :] += F.embedding(codes[..., branch, scale], codebooks[branch, scale])
    return out
