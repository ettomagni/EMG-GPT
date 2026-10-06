"""Checkpoint-compatible canonical EMG-GPT components for pose inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .codebooks import dequantize_tokens

HeadMode = Literal["codebook_depth_ar"]
PoolMode = Literal["cls"]
ScaleWeighting = Literal["uniform", "inverse"]
FrameInputMode = Literal["rvq_sum"]
RVQInputNormalization = Literal["layernorm"]
LatentLossMode = Literal["mse", "smooth_l1", "smooth_l1_cosine"]
TemporalBackbone = Literal["frame_gpt"]


@dataclass(frozen=True)
class EMGGPTConfig:
    """Checkpoint configuration; legacy objective fields are retained as metadata."""

    codebook_size: int
    n_channels: int
    n_branches: int
    n_scales: int
    context_frames: int
    tokenization_context: str = "independent_patch"
    tokenizer_input_patches: int = 1
    temporal_embedding_policy: str = "right_aligned_last"
    patch_size_target_fs: int = 200
    frame_hop_target_fs: int = 40
    token_timestamp_policy: str = "window_end"
    token_time_offset_target_fs: int = 200
    source_fs: int = 2000
    target_fs: int = 1000
    preprocessing_mode: str = "butterworth3_sosfiltfilt_resample_poly_float32_v1"
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1
    temporal_backbone: TemporalBackbone = "frame_gpt"
    head_mode: HeadMode = "codebook_depth_ar"
    frame_input_mode: FrameInputMode = "rvq_sum"
    rvq_input_normalization: RVQInputNormalization = "layernorm"
    frame_pool: PoolMode = "cls"
    head_chunk_size: int = 16
    frame_encoder_layers: int = 2
    head_layers: int = 2
    scale_loss_weighting: ScaleWeighting = "uniform"
    code_dim: int = 128
    aux_latent_weight: float = 0.0
    token_ce_weight: float = 1.0
    latent_loss_mode: LatentLossMode = "mse"
    latent_cosine_weight: float = 1.0
    geometry_soft_target_weight: float = 0.0
    geometry_soft_target_topk: int = 32
    geometry_soft_target_mass: float = 0.5
    codebook_logit_scale_max: float = 100.0
    logit_z_loss_weight: float = 0.0

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    def __post_init__(self):
        required = {
            "temporal_backbone": "frame_gpt",
            "head_mode": "codebook_depth_ar",
            "frame_input_mode": "rvq_sum",
            "frame_pool": "cls",
            "rvq_input_normalization": "layernorm",
            "aux_latent_weight": 0.0,
            "geometry_soft_target_weight": 0.0,
        }
        for name, value in required.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"Unsupported inference configuration: {name}={getattr(self, name)!r}"
                )
        for name in (
            "codebook_size",
            "n_channels",
            "n_branches",
            "n_scales",
            "context_frames",
            "d_model",
            "n_layer",
            "n_head",
            "code_dim",
            "frame_encoder_layers",
            "head_layers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention.

    ``causal=True`` (the default) applies a causal mask and is used by the
    temporal GPT backbone. ``causal=False`` is bidirectional and is used by the
    intra-frame spatial encoder and the structured output head, where all
    channel/branch positions inside a single 200 ms frame are simultaneous and
    must see one another.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float, causal: bool = True):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        attn = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.proj(attn)


class CrossAttention(nn.Module):
    """Multi-head attention with queries from ``x`` and keys/values from ``memory``.

    Used by the factorized head so each target channel/branch query can read the
    temporal context vector produced by the GPT backbone.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        bsz, len_q, d_model = x.shape
        len_kv = memory.shape[1]
        query = self.q(x).view(bsz, len_q, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv(memory).view(bsz, len_kv, 2, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        key, value = kv.unbind(dim=0)
        attn = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, len_q, d_model)
        return self.proj(attn)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, causal: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout, causal=causal)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FrameDecoderBlock(nn.Module):
    """One block of the factorized output head.

    Target queries (one per channel/branch position of the frame being
    predicted) first attend to one another (bidirectional self-attention across
    positions), then cross-attend to the temporal context vector, then pass
    through an MLP. The cross-position self-attention is what removes the rank-1
    limit of the additive-bias head: a position's prediction is a nonlinear
    function of the context and of every other position, not a shared vector
    plus a fixed per-position bias.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.ln_sa = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, n_head, dropout, causal=False)
        self.ln_ca = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_head, dropout)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln_sa(x))
        x = x + self.cross_attn(self.ln_ca(x), memory)
        x = x + self.mlp(self.ln_mlp(x))
        return x


class CodebookDistanceHead(nn.Module):
    """Retained prediction-head parameters for strict historical checkpoint loading."""

    def __init__(self, config: EMGGPTConfig):
        super().__init__()
        self.config = config
        self.channel_embed = nn.Embedding(config.n_channels, config.d_model)
        self.branch_embed = nn.Embedding(config.n_branches, config.d_model)
        self.scale_embed = nn.Embedding(config.n_scales, config.d_model)
        self.hidden_proj = nn.Linear(config.d_model, config.d_model)
        self.blocks = nn.ModuleList(
            [
                FrameDecoderBlock(config.d_model, config.n_head, config.dropout)
                for _ in range(config.head_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.d_model)
        self.query_proj = nn.Linear(config.d_model, config.code_dim)
        initial_scale = math.log(math.sqrt(config.code_dim))
        self.logit_scale = nn.Parameter(
            torch.full((config.n_branches, config.n_scales), initial_scale)
        )
        self.uses_depth_context = config.head_mode == "codebook_depth_ar"
        if self.uses_depth_context:
            self.depth_norm = nn.LayerNorm(config.code_dim)
            self.depth_proj = nn.Linear(config.code_dim, config.d_model, bias=False)
        else:
            self.depth_norm = None
            self.depth_proj = None


class StructuredFrameEmbedding(nn.Module):
    """Sum RVQ latents, encode channel/branch positions and pool the CLS token."""

    def __init__(self, config: EMGGPTConfig):
        super().__init__()
        self.config = config
        self.rvq_input_norm = nn.LayerNorm(config.code_dim)
        self.rvq_input_proj = nn.Linear(config.code_dim, config.d_model, bias=False)
        self.channel_embed = nn.Embedding(config.n_channels, config.d_model)
        self.branch_embed = nn.Embedding(config.n_branches, config.d_model)
        self.time_embed = nn.Embedding(config.context_frames, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.frame_cls = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.frame_cls, mean=0.0, std=0.02)
        self.encoder_blocks = nn.ModuleList(
            [
                GPTBlock(config.d_model, config.n_head, config.dropout, causal=False)
                for _ in range(config.frame_encoder_layers)
            ]
        )
        self.encoder_ln = nn.LayerNorm(config.d_model)

    def forward(self, tokens: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5 or tokens.dtype != torch.int64:
            raise ValueError("Expected int64 tokens [batch,time,channels,branches,scales]")
        bsz, seq_len, channels, branches, scales = tokens.shape
        expected_grid = (self.config.n_channels, self.config.n_branches, self.config.n_scales)
        if (
            channels,
            branches,
            scales,
        ) != expected_grid or not 0 < seq_len <= self.config.context_frames:
            raise ValueError("Token shape does not match the model configuration")
        if tokens.numel() == 0 or tokens.min() < 0 or tokens.max() >= self.config.codebook_size:
            raise ValueError("Token IDs are outside the configured vocabulary")
        channel_ids = torch.arange(channels, device=tokens.device)
        branch_ids = torch.arange(branches, device=tokens.device)
        latent = dequantize_tokens(tokens, codebooks)
        code = self.rvq_input_proj(self.rvq_input_norm(latent))
        code = code + self.channel_embed(channel_ids).view(1, 1, channels, 1, -1)
        code = code + self.branch_embed(branch_ids).view(1, 1, 1, branches, -1)
        flat = code.reshape(bsz * seq_len, channels * branches, self.config.d_model)
        cls = self.frame_cls.view(1, 1, -1).expand(bsz * seq_len, 1, -1)
        encoded = torch.cat([cls, flat], dim=1)
        for block in self.encoder_blocks:
            encoded = block(encoded)
        encoded = self.encoder_ln(encoded)
        frame = encoded[:, 0].view(bsz, seq_len, self.config.d_model)
        time_ids = torch.arange(seq_len, device=tokens.device)
        return self.dropout(frame + self.time_embed(time_ids).view(1, seq_len, -1))


class EMGFrameGPT(nn.Module):
    """Canonical frame encoder and causal GPT; the pretrained head is retained for strict loading."""

    def __init__(self, config: EMGGPTConfig):
        super().__init__()
        self.config = config
        self.frame_embed = StructuredFrameEmbedding(config)
        self.blocks = nn.ModuleList(
            [GPTBlock(config.d_model, config.n_head, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.codebook_head = CodebookDistanceHead(config)
        self.register_buffer(
            "codebooks",
            torch.zeros(config.n_branches, config.n_scales, config.codebook_size, config.code_dim),
            persistent=False,
        )
        self._codebooks_ready = False

    @property
    def requires_codebooks(self):
        return True

    def set_codebooks(self, codebooks: torch.Tensor) -> None:
        if codebooks.shape != self.codebooks.shape or codebooks.dtype != self.codebooks.dtype:
            raise ValueError("Codebook shape/dtype differs from the checkpoint configuration")
        if not torch.isfinite(codebooks).all():
            raise ValueError("Codebooks contain non-finite values")
        self.codebooks.copy_(codebooks.to(self.codebooks.device))
        self._codebooks_ready = True

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self._codebooks_ready:
            raise RuntimeError("Load verified codebooks before inference")
        hidden = self.frame_embed(tokens, self.codebooks)
        for block in self.blocks:
            hidden = block(hidden)
        return self.ln_f(hidden)
