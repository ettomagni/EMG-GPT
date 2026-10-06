"""Capture golden outputs using the original research checkout, never this package.

Run in the research environment with --research-root, --tokenizer, --codebooks,
--regression, --tracking and --output (an empty directory). Checkpoint hashes
must match the published artifacts. This script is not needed to run the tests.
"""

import argparse
import gc
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch
from reference_input import synthetic_boundary, synthetic_emg


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("research-root", "tokenizer", "codebooks", "regression", "tracking", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("--output must be empty")
    sys.path.insert(0, str(args.research_root.resolve()))
    from emg_pose.model import load_pose_model
    from scripts.tokenize_emg2pose import (
        load_tokenizer,
        make_overlapping_blocks,
        preprocess_emg_array,
        tokenize_blocks,
    )

    catalog = json.loads(
        (Path(__file__).parents[1] / "src/emg_gpt/resources/artifacts.json").read_text()
    )
    assert sha(args.tokenizer) == catalog["tokenizer"]["sha256"]
    torch.set_num_threads(1)
    raw = synthetic_emg()
    processed = preprocess_emg_array(raw, source_fs=2000, target_fs=1000, highpass=20, lowpass=400)
    tokenizer, cfg = load_tokenizer(
        args.research_root / "flags/NeuroRVQ_EMG_v1.yml", args.tokenizer, torch.device("cpu")
    )
    assert cfg["n_patches"] == tokenizer.encoder.time_embed.shape[0] == 256
    blocks, _ = make_overlapping_blocks(
        processed, patch_size=200, frame_hop=40, frames_per_block=16
    )
    tokens = tokenize_blocks(
        tokenizer,
        blocks[:10],
        batch_size=16,
        n_time=16,
        patch_size=200,
        max_n_patches=cfg["n_patches"],
        rvq_scales=4,
        device=torch.device("cpu"),
        tokenization_context="independent_patch",
    ).reshape(160, 16, 4, 4)
    values = {"tokens": tokens}
    del tokenizer
    gc.collect()
    for task in ("regression", "tracking"):
        checkpoint = getattr(args, task)
        assert sha(checkpoint) == catalog["pose_models"][task]["source_checkpoint_sha256"]
        model, _ = load_pose_model(checkpoint, codebook_path=args.codebooks)
        model.eval()
        boundary = torch.from_numpy(synthetic_boundary()) if task == "tracking" else None
        with torch.inference_mode():
            values[task] = model(
                torch.from_numpy(tokens[:150].astype(np.int64))[None], initial_pose=boundary
            ).numpy()
        del model
        gc.collect()
    np.savez_compressed(args.output / "reference.npz", **values)
    source_names = [
        "scripts/tokenize_emg2pose.py",
        "flags/NeuroRVQ_EMG_v1.yml",
        "emg_pose/model.py",
        "emg_gpt/model.py",
        "NeuroRVQ_EMG/NeuroRVQ.py",
        "NeuroRVQ_EMG/NeuroRVQ_modules.py",
        "NeuroRVQ_EMG/RVQ.py",
        "NeuroRVQ_EMG/norm_ema_quantizer.py",
    ]
    metadata = {
        "description": "Original research implementation, synthetic EMG, one window per task; not an accuracy benchmark.",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "device": "cpu",
            "threads": 1,
            "token_batch_size": 16,
        },
        "input_sha256": hashlib.sha256(raw.astype("<f4").tobytes()).hexdigest(),
        "reference_sha256": sha(args.output / "reference.npz"),
        "source_sha256": {name: sha(args.research_root / name) for name in source_names},
        "checkpoint_sha256": {
            task: sha(getattr(args, task)) for task in ("regression", "tracking")
        },
        "tokenizer_sha256": sha(args.tokenizer),
        "codebooks_sha256": sha(args.codebooks),
    }
    (args.output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
