# Adapted for EMG-GPT; provenance and license texts are listed in THIRD_PARTY_NOTICES.md.
"""
Residual Vector Quantization Implementation.
Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
"""

import torch
from torch import nn

from .norm_ema_quantizer import NormEMAVectorQuantizer


class ResidualVectorQuantization(nn.Module):
    """Frozen residual quantization with the original state-dict layout."""

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [NormEMAVectorQuantizer(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x):
        quantized_out = torch.zeros_like(x)
        residual = x
        indices = []
        for layer in self.layers:
            quantized, _, codes = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            indices.append(codes)
        return quantized_out, torch.stack(indices), None, None
