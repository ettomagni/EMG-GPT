"""File input and command-line entry points for the public inference API."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .artifacts import fetch_tokenizer
from .inference import PosePredictor
from .tokenization import RAW_CHANNELS


def read_emg(path: Path) -> tuple[np.ndarray, float, tuple[str, ...]]:
    """Read explicit NPZ metadata or the official emg2pose HDF5 signal only."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            missing = {"emg", "sampling_rate_hz", "channel_names"} - set(data.files)
            if missing:
                raise ValueError(f"Input NPZ lacks: {', '.join(sorted(missing))}")
            rate = data["sampling_rate_hz"]
            channels = data["channel_names"]
            if rate.shape != () or rate.dtype.kind not in "fiu":
                raise ValueError("sampling_rate_hz must be a numeric scalar")
            if channels.shape != (16,) or channels.dtype.kind != "U":
                raise ValueError("channel_names must be 16 Unicode strings")
            return data["emg"], float(rate), tuple(channels.tolist())
    if path.suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as error:
            raise ValueError("HDF5 input requires: pip install 'emg-gpt[data]'") from error
        with h5py.File(path, "r") as data:
            group = data["emg2pose"]
            rate = float(group.attrs["sample_rate"])
            # Selecting the compound field avoids loading joint angles or using target poses.
            raw = group["timeseries"].fields("emg")[:]
            times = group["timeseries"].fields("time")[:]
        if times.ndim != 1 or len(times) != len(raw) or not np.isfinite(times).all():
            raise ValueError("Invalid emg2pose timestamps")
        if rate != 2000:
            raise ValueError("HDF5 sample_rate must be 2000 Hz")
        if len(times) > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("HDF5 timestamps must increase strictly")
        # Official recordings contain clock jitter. Preserve the historical nominal
        # sample-index alignment instead of resampling from acquisition timestamps.
        return raw, rate, RAW_CHANNELS
    raise ValueError("Input must be an NPZ or an official emg2pose HDF5 file")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline EMG-GPT hand-pose inference")
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Complete task-specific pose bundle"
    )
    parser.add_argument(
        "--tokenizer", type=Path, required=True, help="Pinned NeuroRVQ tokenizer checkpoint"
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw 2-kHz NPZ or emg2pose HDF5")
    parser.add_argument("--output", type=Path, required=True, help="New prediction .npz file")
    parser.add_argument(
        "--boundary-poses",
        type=Path,
        help="Tracking-only NPZ: initial_poses_rad, boundary_timestamps_s",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--token-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-windows", type=int, help="Limit decoded windows; still filter the full recording"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        if args.output.suffix != ".npz":
            raise ValueError("Prediction output must end in .npz")
        raw, rate, channels = read_emg(args.input)
        boundary = {}
        if args.boundary_poses is not None:
            with np.load(args.boundary_poses, allow_pickle=False) as data:
                boundary = {
                    key: data[key] for key in ("initial_poses_rad", "boundary_timestamps_s")
                }
        predictor = PosePredictor(
            args.model_dir,
            args.tokenizer,
            device=args.device,
            token_batch_size=args.token_batch_size,
        )
        prediction = predictor.predict(
            raw,
            sampling_rate_hz=rate,
            channel_names=channels,
            max_windows=args.max_windows,
            **boundary,
        )
        prediction.save(args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"emg-gpt-predict: {error}\n")
    print(f"Saved {len(prediction.joint_angles_rad)} {predictor.task} windows to {args.output}")


def download_tokenizer(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned NeuroRVQ tokenizer"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(fetch_tokenizer(args.output))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"emg-gpt-download-tokenizer: {error}\n")


if __name__ == "__main__":
    main()
