"""Original offline EMG frontend and frozen independent-patch tokenization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import signal

from .artifacts import resource_json, verify_file
from .bundle import _strict_load
from .neurorvq.channels import GLOBAL_EMG_CHANNELS
from .neurorvq.NeuroRVQ import NeuroRVQTokenizer
from .neurorvq.NeuroRVQ_modules import get_encoder_decoder_params

RAW_CHANNELS = tuple(f"c{i}" for i in range(1, 17))
TOKEN_CHANNELS = tuple(c.decode() for c in GLOBAL_EMG_CHANNELS)


def load_tokenizer(path: Path | str, device: torch.device) -> NeuroRVQTokenizer:
    """Load only the verified upstream tensor checkpoint, including its original topology."""
    path = Path(path)
    verify_file(path, resource_json("artifacts.json")["tokenizer"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Tokenizer must be a tensor-only state dictionary")
    config = resource_json("tokenizer.json")
    encoder, decoder = get_encoder_decoder_params(config)
    tokenizer = NeuroRVQTokenizer(
        encoder,
        decoder,
        n_code=config["n_code"],
        code_dim=config["code_dim"],
        decoder_out_dim=config["decoder_out_dim"],
    )
    _strict_load(tokenizer, state, name="NeuroRVQ tokenizer")
    for branch in range(1, 5):
        for layer in getattr(tokenizer, f"quantize_{branch}").layers:
            if not bool(layer.embedding.initted.item()):
                raise ValueError("Tokenizer contains an uninitialized codebook")
    return tokenizer.to(device).eval().requires_grad_(False)


def preprocess_emg(raw_emg: np.ndarray) -> np.ndarray:
    """Filter the complete recording; return [canonical channels, 1-kHz samples]."""
    raw_emg = np.asarray(raw_emg, dtype=np.float32)
    if raw_emg.ndim != 2 or raw_emg.shape[1] != 16 or len(raw_emg) < 22:
        raise ValueError("Expected raw EMG [samples,16] with at least 22 samples")
    if not np.isfinite(raw_emg).all():
        raise ValueError("EMG contains NaN or infinity")
    sos = signal.butter(3, [20.0, 399.5], btype="bandpass", fs=2000, output="sos")
    filtered = signal.sosfiltfilt(sos, raw_emg, axis=0).astype(np.float32)
    resampled = signal.resample_poly(filtered, up=1, down=2, axis=0).astype(np.float32)
    order = [RAW_CHANNELS.index(channel) for channel in TOKEN_CHANNELS]
    return resampled[:, order].T


@torch.inference_mode()
def tokenize_frames(
    tokenizer: NeuroRVQTokenizer,
    emg: np.ndarray,
    n_frames: int,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Encode complete 200-sample patches on the original 40-sample grid."""
    if type(batch_size) is not int or batch_size < 1 or n_frames < 1:
        raise ValueError("Tokenization needs positive batch_size and n_frames")
    if emg.ndim != 2 or emg.shape[0] != 16 or (n_frames - 1) * 40 + 200 > emg.shape[1]:
        raise ValueError("Requested token grid exceeds the preprocessed recording")
    output = np.empty((n_frames, 16, 4, 4), dtype=np.uint16)
    tokenizer.eval()
    for start in range(0, n_frames, batch_size):
        stop = min(start + batch_size, n_frames)
        patches = np.stack([emg[:, i * 40 : i * 40 + 200] for i in range(start, stop)])
        x = torch.from_numpy(patches).to(device=device, dtype=torch.float32).unsqueeze(2)
        # Match training: the last learned position is 255, independent of storage-block size.
        temporal = torch.full(
            (len(x), 16),
            tokenizer.encoder.time_embed.shape[0] - 1,
            device=device,
            dtype=torch.long,
        )
        spatial = torch.arange(16, device=device).expand(len(x), -1)
        _, indices, _, _ = tokenizer.encode(x, temporal, spatial)
        branches = [q[:4].reshape(4, len(x), 16, 1) for q in indices]
        codes = torch.stack(branches).permute(2, 4, 3, 0, 1).contiguous()
        output[start:stop] = codes[:, 0].cpu().numpy().astype(np.uint16)
    return output
