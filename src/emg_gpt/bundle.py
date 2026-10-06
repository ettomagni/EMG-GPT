"""Strict, tensor-only loading of complete adapted pose bundles."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from .artifacts import resource_json, sha256, verify_file
from .model import EMGFrameGPT, EMGGPTConfig
from .pose import EMGPoseModel, PoseModelConfig

_SHA256 = re.compile(r"[0-9a-f]{64}")
_POSE_TASKS = {"regression_lstm": "regression", "tracking_lstm": "tracking"}


def validate_pose_evaluation_contract(
    pose_config: object, contract: object
) -> tuple[str, dict[str, int]]:
    """Validate the task-specific public pose evaluation contract."""
    if not isinstance(pose_config, dict) or not isinstance(contract, dict):
        raise ValueError("Pose config and evaluation contract must be mappings")
    head_mode = pose_config.get("head_mode")
    task = _POSE_TASKS.get(head_mode)
    if task is None:
        raise ValueError(f"Unsupported public pose head mode: {head_mode!r}")
    fields = ("context_frames", "score_tail_frames", "pose_hz", "pose_offset_frames")
    invalid = [name for name in fields if type(contract.get(name)) is not int]
    if invalid:
        raise ValueError("Pose evaluation contract requires integer fields: " + ", ".join(invalid))
    normalized = {name: int(contract[name]) for name in fields}
    context_frames = normalized["context_frames"]
    score_tail_frames = normalized["score_tail_frames"]
    if context_frames < 1 or not 0 < score_tail_frames < context_frames:
        raise ValueError("Pose evaluation contract requires 0 < score_tail_frames < context_frames")
    if normalized["pose_hz"] < 1:
        raise ValueError("Pose evaluation contract requires pose_hz > 0")
    expected_offset = 0 if task == "regression" else 1
    if normalized["pose_offset_frames"] != expected_offset:
        raise ValueError(f"{task} requires pose_offset_frames={expected_offset}")
    left_context_field = f"{task}_left_context_frames"
    expected_left_context = context_frames - score_tail_frames
    if pose_config.get(left_context_field) != expected_left_context:
        raise ValueError(f"Pose config {left_context_field} does not match the evaluation contract")
    if (
        type(pose_config.get("pose_steps_per_frame")) is not int
        or int(pose_config["pose_steps_per_frame"]) < 1
    ):
        raise ValueError("Pose config requires a positive integer pose_steps_per_frame")
    return (task, normalized)


def _verify_bundle(root: Path, *, require_pinned: bool) -> tuple[dict, bool]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported bundle manifest: {manifest_path}")
    if manifest.get("kind") != "pose-model":
        raise ValueError(f"Not a pose-model manifest: {manifest_path}")
    source_sha256 = manifest.get("source", {}).get("sha256")
    if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
        raise ValueError(f"Bundle manifest lacks a valid source SHA-256: {manifest_path}")
    # Official bundles use the trusted catalog; other bundles use their own manifest.
    for record in resource_json("artifacts.json")["pose_models"].values():
        if record["source_checkpoint_sha256"] == source_sha256:
            for name, identity in record["files"].items():
                verify_file(root / name, identity)
            return manifest, True
    if require_pinned:
        raise ValueError("Model is not a pinned EMG-GPT release bundle")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Bundle manifest lacks outputs: {manifest_path}")
    for name in ("config.json", "model.safetensors", "codebooks.safetensors"):
        record = outputs.get(name)
        path = root / name
        if not isinstance(record, dict) or not path.is_file():
            raise FileNotFoundError(f"Bundle is missing verified output {name!r}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"Bundle byte-size mismatch for {name}")
        if sha256(path) != record.get("sha256"):
            raise ValueError(f"Bundle SHA-256 mismatch for {name}")
    return manifest, False


def _strict_load(module: nn.Module, tensors: dict[str, torch.Tensor], *, name: str) -> None:
    expected = module.state_dict()
    if set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        unexpected = sorted(set(tensors) - set(expected))
        raise ValueError(
            f"{name} tensor keys are incompatible: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for key, tensor in tensors.items():
        reference = expected[key]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(
                f"{name} tensor metadata is incompatible for {key}: bundle=({tensor.dtype}, {tuple(tensor.shape)}), model=({reference.dtype}, {tuple(reference.shape)})"
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite tensor {key}")
    module.load_state_dict(tensors, strict=True)


def load_pose_model_bundle(
    directory: Path | str, *, device: str | torch.device = "cpu", require_pinned: bool = False
) -> EMGPoseModel:
    """Check bundle integrity; optionally require identity against the release catalog."""
    root = Path(directory)
    manifest, pinned = _verify_bundle(root, require_pinned=require_pinned)
    document = json.loads((root / "config.json").read_text())
    if document.get("schema_version") != 1 or document.get("model_type") != "emg_gpt_pose_model":
        raise ValueError("Not a complete EMG-GPT pose-model bundle")
    source_sha256 = manifest.get("source", {}).get("sha256")
    if document.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("Pose config and manifest identify different source checkpoints")
    source_gpt_sha256 = document.get("source_gpt_checkpoint_sha256")
    if not isinstance(source_gpt_sha256, str) or _SHA256.fullmatch(source_gpt_sha256) is None:
        raise ValueError("Complete pose bundle lacks a valid source GPT SHA-256")
    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("Complete pose bundle lacks dependency provenance")
    codebook_dependency = dependencies.get("codebooks")
    initial_dependency = dependencies.get("initial_model")
    if not isinstance(codebook_dependency, dict) or not isinstance(initial_dependency, dict):
        raise ValueError("Complete pose bundle lacks codebook or initial-model provenance")
    codebook_sha256 = codebook_dependency.get("sha256")
    if (
        not isinstance(codebook_sha256, str)
        or _SHA256.fullmatch(codebook_sha256) is None
        or document.get("codebook_source_sha256") != codebook_sha256
    ):
        raise ValueError("Pose config and manifest identify different codebooks")
    initial_model = document.get("initial_model")
    if not isinstance(initial_model, dict) or initial_model != {
        "kind": initial_dependency.get("kind"),
        "sha256": initial_dependency.get("sha256"),
    }:
        raise ValueError("Pose config and manifest identify different initial models")
    if (
        initial_model.get("kind")
        not in {"gpt_checkpoint", "warm_start_pose_checkpoint", "resume_pose_checkpoint"}
        or not isinstance(initial_model.get("sha256"), str)
        or _SHA256.fullmatch(initial_model["sha256"]) is None
    ):
        raise ValueError("Complete pose bundle has invalid initial-model provenance")
    pose_values = document.get("pose_config")
    backbone_values = document.get("backbone_model_config")
    if not isinstance(pose_values, dict) or not isinstance(backbone_values, dict):
        raise ValueError("Complete pose bundle lacks model configuration")
    pose_config = PoseModelConfig(**pose_values)
    if pose_config.backbone_mode not in ("top", "full"):
        raise ValueError("Complete pose bundle requires a top/full adapted backbone")
    expected_task, _ = validate_pose_evaluation_contract(
        pose_values, document.get("evaluation_contract")
    )
    if document.get("task") != expected_task:
        raise ValueError("Pose bundle task and head mode are inconsistent")
    backbone = EMGFrameGPT(EMGGPTConfig(**backbone_values))
    model = EMGPoseModel(backbone, pose_config)
    _strict_load(
        model,
        load_file(str(root / "model.safetensors"), device="cpu"),
        name="Complete pose bundle",
    )
    codebook_payload = load_file(str(root / "codebooks.safetensors"), device="cpu")
    if set(codebook_payload) != {"codebooks"}:
        raise ValueError("Codebook bundle must contain exactly the 'codebooks' tensor")
    backbone.set_codebooks(codebook_payload["codebooks"])
    model._bundle_pinned = pinned
    return model.to(device).eval().requires_grad_(False)
