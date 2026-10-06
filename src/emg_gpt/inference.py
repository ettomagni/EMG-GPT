"""Raw EMG to independently decoded, timestamped pose windows."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import resource_json, sha256
from .hub import load_pose_model_bundle
from .tokenization import RAW_CHANNELS, load_tokenizer, preprocess_emg, tokenize_frames

# Names/index order from the pinned emg2pose joint convention; no kinematics dependency.
JOINT_NAMES = (
    "THUMB_CMC_FE",
    "THUMB_CMC_AA",
    "THUMB_MCP_FE",
    "THUMB_IP_FE",
    "INDEX_MCP_AA",
    "INDEX_MCP_FE",
    "INDEX_PIP_FE",
    "INDEX_DIP_FE",
    "MIDDLE_MCP_AA",
    "MIDDLE_MCP_FE",
    "MIDDLE_PIP_FE",
    "MIDDLE_DIP_FE",
    "RING_MCP_AA",
    "RING_MCP_FE",
    "RING_PIP_FE",
    "RING_DIP_FE",
    "PINKY_MCP_AA",
    "PINKY_MCP_FE",
    "PINKY_PIP_FE",
    "PINKY_DIP_FE",
)


@dataclass(frozen=True)
class WindowPlan:
    """Canonical storage truncation, window starts and raw-sample alignment."""

    token_starts: np.ndarray
    source_sample_indices: np.ndarray
    stored_token_frames: int
    total_windows: int

    @property
    def timestamps_s(self) -> np.ndarray:
        return self.source_sample_indices / 2000.0

    @property
    def boundary_timestamps_s(self) -> np.ndarray:
        return self.timestamps_s[:, 0]


def plan_windows(n_samples: int, task: str, *, max_windows: int | None = None) -> WindowPlan:
    """Match the original token-pose dataset index, without reading any target poses."""
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if task not in ("regression", "tracking"):
        raise ValueError("task must be regression or tracking")
    if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
        raise ValueError("max_windows must be a positive integer")
    target_samples = (n_samples + 1) // 2
    n_frames = max(0, 1 + (target_samples - 200) // 40)
    stored = n_frames // 16 * 16
    offset = int(task == "tracking")
    stride = 125 if offset else 150
    # The historical dataset bounds the full 300-pose-sample context, including its tail.
    max_raw_start = (n_samples - 1 - 299 * 40 - (200 + offset * 40) * 2) // 80
    last_start = min(stored - 150 - offset, max_raw_start)
    if last_start < 0:
        raise ValueError(
            "Recording is too short for a complete canonical window (minimum 13,119 samples at 2 kHz)"
        )
    starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
    total = len(starts)
    if max_windows is not None:
        starts = starts[:max_windows]
    samples = 400 + (starts[:, None] + 25 + offset) * 80 + np.arange(250) * 40
    return WindowPlan(starts, samples, stored, total)


@dataclass(frozen=True)
class PosePrediction:
    """Pose radians [windows,250,20], with gaps represented by the coverage mask."""

    joint_angles_rad: np.ndarray
    source_sample_indices: np.ndarray
    coverage_mask: np.ndarray
    window_token_starts: np.ndarray
    metadata: dict

    @property
    def timestamps_s(self) -> np.ndarray:
        return self.source_sample_indices / 2000.0

    def save(self, destination: Path | str) -> None:
        """Write a pickle-free NPZ atomically, refusing to overwrite an existing file."""
        destination = Path(destination)
        if destination.suffix != ".npz":
            raise ValueError("Prediction output must end in .npz")
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            staged = Path(handle.name)
        try:
            with staged.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    joint_angles_rad=self.joint_angles_rad,
                    timestamps_s=self.timestamps_s,
                    source_sample_indices=self.source_sample_indices,
                    window_token_starts=self.window_token_starts,
                    joint_names=np.asarray(JOINT_NAMES),
                    valid=np.isfinite(self.joint_angles_rad).all(axis=-1),
                    coverage_mask=self.coverage_mask,
                    coverage_timestamps_s=np.arange(len(self.coverage_mask)) / 50.0,
                    metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
                )
            os.link(staged, destination)
        finally:
            staged.unlink(missing_ok=True)


def _validate_frontend(model, task: str, contract: dict) -> None:
    expected = {
        "n_channels": 16,
        "n_branches": 4,
        "n_scales": 4,
        "context_frames": 150,
        "codebook_size": 8192,
        "code_dim": 128,
        "source_fs": 2000,
        "target_fs": 1000,
        "patch_size_target_fs": 200,
        "frame_hop_target_fs": 40,
        "token_timestamp_policy": "window_end",
        "token_time_offset_target_fs": 200,
        "tokenization_context": "independent_patch",
        "tokenizer_input_patches": 1,
        "temporal_embedding_policy": "right_aligned_last",
        "preprocessing_mode": "butterworth3_sosfiltfilt_resample_poly_float32_v1",
    }
    for key, value in expected.items():
        if getattr(model.backbone.config, key) != value:
            raise ValueError(f"Incompatible raw-EMG frontend: {key}")
    expected_contract = {
        "context_frames": 150,
        "score_tail_frames": 125,
        "pose_hz": 50,
        "pose_offset_frames": int(task == "tracking"),
    }
    if (
        contract != expected_contract
        or model.config.n_joints != 20
        or model.config.pose_steps_per_frame != 2
    ):
        raise ValueError(
            "Bundle does not implement the canonical 150/125-frame, 50-Hz pose contract"
        )


class PosePredictor:
    """Offline inference using a complete pose bundle and the pinned NeuroRVQ tokenizer."""

    def __init__(
        self,
        model_dir: Path | str,
        tokenizer_path: Path | str,
        *,
        device: str = "cpu",
        token_batch_size: int = 16,
    ):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda; MPS is not validated")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if type(token_batch_size) is not int or token_batch_size < 1:
            raise ValueError("token_batch_size must be a positive integer")
        self.device = torch.device(device)
        self.token_batch_size = token_batch_size
        self.model = load_pose_model_bundle(model_dir, device=self.device)
        document = json.loads((Path(model_dir) / "config.json").read_text())
        self.task = document["task"]
        _validate_frontend(self.model, self.task, document["evaluation_contract"])
        self.tokenizer = load_tokenizer(tokenizer_path, self.device)
        for branch in range(4):
            for scale in range(4):
                actual = (
                    getattr(self.tokenizer, f"quantize_{branch + 1}").layers[scale].embedding.weight
                )
                if not torch.equal(actual, self.model.backbone.codebooks[branch, scale]):
                    raise ValueError("Pose bundle codebooks differ from the verified tokenizer")
        self.identity = {
            "pose_source_checkpoint_sha256": document["source_checkpoint_sha256"],
            "pose_bundle_manifest_sha256": sha256(Path(model_dir) / "manifest.json"),
            "tokenizer_sha256": resource_json("artifacts.json")["tokenizer"]["sha256"],
        }

    def plan(self, n_samples: int, *, max_windows: int | None = None) -> WindowPlan:
        return plan_windows(n_samples, self.task, max_windows=max_windows)

    @torch.inference_mode()
    def predict(
        self,
        raw_emg: np.ndarray,
        *,
        sampling_rate_hz: int,
        channel_names: Sequence[str],
        initial_poses_rad: np.ndarray | None = None,
        boundary_timestamps_s: np.ndarray | None = None,
        max_windows: int | None = None,
    ) -> PosePrediction:
        """Predict independent windows; Tracking takes one measured boundary pose per window."""
        if sampling_rate_hz != 2000 or isinstance(sampling_rate_hz, bool):
            raise ValueError("Raw EMG must be sampled at 2000 Hz; do not pre-resample it")
        raw = np.asarray(raw_emg)
        if raw.ndim != 2 or raw.shape[1] != 16 or raw.dtype.kind not in "fiu":
            raise ValueError("raw_emg must be a real numeric [samples,16] array")
        if not np.isfinite(raw).all():
            raise ValueError("raw_emg contains NaN or infinity")
        names = tuple(channel_names)
        if len(names) != 16 or set(names) != set(RAW_CHANNELS):
            raise ValueError("channel_names must contain c1 through c16 exactly once")
        plan = self.plan(len(raw), max_windows=max_windows)
        initial = None
        if self.task == "tracking":
            if initial_poses_rad is None or boundary_timestamps_s is None:
                raise ValueError(
                    "Tracking requires initial_poses_rad and boundary_timestamps_s for every window"
                )
            initial = np.asarray(initial_poses_rad)
            times = np.asarray(boundary_timestamps_s)
            if (
                initial.dtype.kind not in "fiu"
                or initial.shape != (len(plan.token_starts), 20)
                or not np.isfinite(initial).all()
            ):
                raise ValueError("initial_poses_rad must be finite [windows,20] radians")
            if (
                times.dtype.kind not in "fiu"
                or times.shape != (len(initial),)
                or not np.isfinite(times).all()
                or not np.allclose(times, plan.boundary_timestamps_s, rtol=0, atol=1e-9)
            ):
                raise ValueError(
                    "Boundary timestamps must match predictor.plan(...).boundary_timestamps_s"
                )
        elif initial_poses_rad is not None or boundary_timestamps_s is not None:
            raise ValueError("Regression does not accept boundary poses")
        raw = raw[:, [names.index(c) for c in RAW_CHANNELS]]
        processed = preprocess_emg(raw)
        # Preserve whole-record filtering even when only a prefix of windows is requested.
        needed = int(plan.token_starts[-1]) + 150
        frame_count = (needed + 15) // 16 * 16
        tokens = tokenize_frames(
            self.tokenizer,
            processed,
            frame_count,
            batch_size=self.token_batch_size,
            device=self.device,
        )
        self.model.eval()
        predictions = []
        for index, start in enumerate(plan.token_starts):
            window = (
                torch.from_numpy(tokens[start : start + 150].astype(np.int64))
                .unsqueeze(0)
                .to(self.device)
            )
            boundary = (
                None
                if initial is None
                else torch.as_tensor(
                    initial[index : index + 1], dtype=torch.float32, device=self.device
                )
            )
            pose = self.model(window, initial_pose=boundary).cpu().numpy()[0]
            if pose.shape != (250, 20) or not np.isfinite(pose).all():
                raise ValueError("Model produced invalid pose predictions")
            predictions.append(pose)
        angles = np.stack(predictions)
        coverage = np.zeros((len(raw) + 39) // 40, dtype=bool)
        coverage[plan.source_sample_indices.reshape(-1) // 40] = True
        metadata = {
            "schema_version": 1,
            "task": self.task,
            "pose_units": "radians",
            "input_units": "emg2pose_native_scale",
            "sample_rate_hz": 2000,
            "pose_rate_hz": 50,
            "input_samples": len(raw),
            "input_channel_order": list(names),
            "preprocessing": "whole_record_zero_phase_20_399.5Hz_resample_poly_2000_to_1000",
            "context_token_frames": 150,
            "scored_token_frames": 125,
            "stride_token_frames": 125 if self.task == "tracking" else 150,
            "token_timestamp": "patch_end",
            "pose_offset_token_frames": int(self.task == "tracking"),
            "stored_token_frames": plan.stored_token_frames,
            "total_available_windows": plan.total_windows,
            "predicted_windows": len(angles),
            "window_state": "reset_independently",
            "boundary_pose_source": "explicit_caller_input" if initial is not None else "none",
            "boundary_timestamps_s": plan.boundary_timestamps_s.tolist()
            if initial is not None
            else [],
            "device": str(self.device),
            "dtype": "float32",
            **self.identity,
        }
        return PosePrediction(
            angles, plan.source_sample_indices, coverage, plan.token_starts, metadata
        )
