# Adapted for EMG-GPT; provenance and license texts are listed in THIRD_PARTY_NOTICES.md.
"""
This file is based on LaBraM, BEiT-v2, timm, DeiT, and DINO code bases
https://github.com/935963004/LaBraM/blob/main/norm_ema_quantizer.py
https://github.com/microsoft/unilm/tree/master/beitv2
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit/
https://github.com/facebookresearch/dino
"""

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class EmbeddingEMA(nn.Module):
    """Frozen embedding, retaining all historical checkpoint tensors."""

    def __init__(self, num_tokens, codebook_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_tokens, codebook_dim), requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(torch.zeros(num_tokens, codebook_dim), requires_grad=False)
        self.register_buffer("initted", torch.zeros(1))

    def forward(self, indices):
        return F.embedding(indices, self.weight)


class NormEMAVectorQuantizer(nn.Module):
    """Original normalized nearest-code arithmetic, without EMA updates or losses."""

    def __init__(
        self,
        n_embed,
        embedding_dim,
        beta,
        decay=0.99,
        eps=1e-5,
        statistic_code_usage=True,
        kmeans_init=False,
        codebook_init_path="",
    ):
        super().__init__()
        if codebook_init_path:
            raise ValueError("Load an initialized tokenizer checkpoint for inference")
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.embedding = EmbeddingEMA(n_embed, embedding_dim)
        if statistic_code_usage:
            self.register_buffer("cluster_size", torch.zeros(n_embed))

    def forward(self, z):
        if not bool(self.embedding.initted.item()):
            raise ValueError("Uninitialized tokenizer codebook")
        z = F.normalize(rearrange(z, "b c h w -> b h w c"), p=2, dim=-1)
        flat = z.reshape(-1, self.codebook_dim)
        distance = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", flat, self.embedding.weight)
        )
        indices = torch.argmin(distance, dim=1)
        quantized = self.embedding(indices).view(z.shape)
        # Keep the historical cancellation arithmetic: it affects later RVQ levels.
        quantized = z + (quantized - z).detach()
        return rearrange(quantized, "b h w c -> b c h w"), None, indices
