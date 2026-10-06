"""Strict, tensor-only loading of complete adapted pose bundles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from torch import nn

from emg_pose.model import EMGPoseModel, PoseModelConfig

from .artifacts import resource_json, verify_file
from .model import EMGFrameGPT, EMGGPTConfig

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_bundle(root: Path, required: tuple[str, ...], *, kind: str) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported bundle manifest: {manifest_path}")
    if manifest.get("kind") != kind:
        raise ValueError(f"Bundle manifest kind is not {kind!r}: {manifest_path}")
    source_sha256 = manifest.get("source", {}).get("sha256")
    if not isinstance(source_sha256, str) or _SHA256.fullmatch(source_sha256) is None:
        raise ValueError(f"Bundle manifest lacks a valid source SHA-256: {manifest_path}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Bundle manifest lacks outputs: {manifest_path}")
    for name in required:
        record = outputs.get(name)
        path = root / name
        if not isinstance(record, dict) or not path.is_file():
            raise FileNotFoundError(f"Bundle is missing verified output {name!r}")
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise ValueError(f"Bundle byte-size mismatch for {name}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"Bundle SHA-256 mismatch for {name}")
    return manifest


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError("Install the required safetensors dependency") from error
    return load_file(str(path), device="cpu")


def _tensor_key_sha256(keys: object) -> str:
    if not isinstance(keys, list) or not keys or (not all(isinstance(key, str) for key in keys)):
        raise ValueError("Bundle tensor key manifest must be a non-empty string list")
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        raise ValueError("Bundle tensor keys must be unique and sorted")
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _load_verified_safetensors(
    root: Path, name: str, manifest: dict, *, require_metadata: bool = False
) -> dict[str, torch.Tensor]:
    record = manifest["outputs"][name]
    expected_keys = record.get("tensor_keys")
    expected_key_sha256 = _tensor_key_sha256(expected_keys)
    if record.get("tensor_key_sha256") != expected_key_sha256:
        raise ValueError(f"Bundle tensor-key SHA-256 mismatch for {name}")
    if record.get("exact_tensor_roundtrip") is not True:
        raise ValueError(f"Bundle does not attest exact tensor round-trip for {name}")
    tensors = _load_safetensors(root / name)
    if sorted(tensors) != expected_keys:
        raise ValueError(f"Bundle tensor keys differ from the manifest for {name}")
    metadata = record.get("tensor_metadata")
    if require_metadata and (not isinstance(metadata, dict)):
        raise ValueError(f"Bundle lacks tensor metadata for {name}")
    if isinstance(metadata, dict):
        if set(metadata) != set(tensors):
            raise ValueError(f"Bundle tensor metadata keys differ for {name}")
        for key, tensor in tensors.items():
            expected = metadata[key]
            actual = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            if not isinstance(expected, dict) or expected != actual:
                raise ValueError(f"Bundle tensor metadata mismatch for {name}:{key}")
    return tensors


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
    directory: Path | str, *, device: str | torch.device = "cpu"
) -> EMGPoseModel:
    """Load a complete adapted model; initializer files are not runtime dependencies."""
    root = Path(directory)
    manifest = _verify_bundle(
        root, ("config.json", "model.safetensors", "codebooks.safetensors"), kind="pose-model"
    )
    document = json.loads((root / "config.json").read_text())
    if document.get("schema_version") != 1 or document.get("model_type") != "emg_gpt_pose_model":
        raise ValueError("Not a complete EMG-GPT pose-model bundle")
    source_sha256 = manifest.get("source", {}).get("sha256")
    for record in resource_json("artifacts.json")["pose_models"].values():
        if record["source_checkpoint_sha256"] == source_sha256 and record["files"]:
            for name, identity in record["files"].items():
                verify_file(root / name, identity)
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
        _load_verified_safetensors(root, "model.safetensors", manifest, require_metadata=True),
        name="Complete pose bundle",
    )
    codebook_payload = _load_verified_safetensors(
        root, "codebooks.safetensors", manifest, require_metadata=True
    )
    if set(codebook_payload) != {"codebooks"}:
        raise ValueError("Codebook bundle must contain exactly the 'codebooks' tensor")
    if backbone.requires_codebooks:
        codebooks = codebook_payload["codebooks"]
        if codebooks.dtype != backbone.codebooks.dtype:
            raise ValueError("Codebook dtype does not match the pose backbone")
        backbone.set_codebooks(codebooks)
    model._bundle_source_sha256 = source_sha256
    return model.to(device).eval().requires_grad_(False)
