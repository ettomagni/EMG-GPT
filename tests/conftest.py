"""Small random tensors test software contracts, never model accuracy."""

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from emg_gpt.artifacts import sha256
from emg_gpt.model import EMGFrameGPT, EMGGPTConfig
from emg_pose.model import EMGPoseModel, PoseModelConfig


def update_record(root, name, tensors=None):
    manifest = json.loads((root / "manifest.json").read_text())
    record = {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size}
    if tensors is not None:
        keys = sorted(tensors)
        record.update(
            tensor_keys=keys,
            tensor_key_sha256=hashlib.sha256("\n".join(keys).encode()).hexdigest(),
            tensor_metadata={
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()
            },
            exact_tensor_roundtrip=True,
        )
    manifest["outputs"][name] = record
    (root / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def bundle_factory(tmp_path):
    def make(task="regression"):
        torch.manual_seed(7)
        cfg = EMGGPTConfig(
            codebook_size=8,
            n_channels=2,
            n_branches=2,
            n_scales=2,
            context_frames=6,
            d_model=8,
            n_layer=1,
            n_head=2,
            code_dim=4,
            frame_encoder_layers=1,
            head_layers=1,
            dropout=0,
        )
        pose_cfg = PoseModelConfig(
            head_mode=f"{task}_lstm",
            head_hidden=8,
            head_layers=1,
            tracking_feature_dim=4,
            regression_feature_dim=4,
            tracking_left_context_frames=2,
            regression_left_context_frames=2,
            backbone_mode="full",
            dropout=0,
        )
        model = EMGPoseModel(EMGFrameGPT(cfg), pose_cfg).eval()
        model.backbone.set_codebooks(torch.randn(2, 2, 8, 4))
        initial = {"kind": "warm_start_pose_checkpoint", "sha256": "a" * 64}
        document = {
            "schema_version": 1,
            "model_type": "emg_gpt_pose_model",
            "task": task,
            "pose_config": pose_cfg.to_dict(),
            "backbone_model_config": cfg.to_dict(),
            "source_checkpoint_sha256": "b" * 64,
            "source_gpt_checkpoint_sha256": "c" * 64,
            "codebook_source_sha256": "d" * 64,
            "initial_model": initial,
            "evaluation_contract": {
                "context_frames": 6,
                "score_tail_frames": 4,
                "pose_hz": 50,
                "pose_offset_frames": int(task == "tracking"),
            },
        }
        root = tmp_path / task
        root.mkdir()
        (root / "config.json").write_text(json.dumps(document))
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pose-model",
                    "source": {"sha256": "b" * 64},
                    "dependencies": {"initial_model": initial, "codebooks": {"sha256": "d" * 64}},
                    "outputs": {},
                }
            )
        )
        update_record(root, "config.json")
        for name, tensors in (
            ("model.safetensors", model.state_dict()),
            ("codebooks.safetensors", {"codebooks": model.backbone.codebooks}),
        ):
            save_file(tensors, root / name)
            update_record(root, name, tensors)
        return root, model

    return make
