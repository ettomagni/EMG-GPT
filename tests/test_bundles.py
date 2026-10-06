import json

import pytest
import torch
from conftest import update_record
from safetensors.torch import load_file, save_file

from emg_gpt.hub import load_pose_model_bundle
from emg_gpt.inference import PosePredictor


@pytest.mark.parametrize("task", ["regression", "tracking"])
def test_complete_bundle_roundtrip_and_reset(bundle_factory, task):
    root, original = bundle_factory(task)
    restored = load_pose_model_bundle(root)
    tokens = torch.randint(0, 8, (1, 6, 2, 2, 2))
    initial = torch.ones(1, 20) if task == "tracking" else None
    with torch.inference_mode():
        reference = original(tokens, initial)
        first = restored(tokens, initial)
        second = restored(tokens, initial)
    assert first.shape == (1, 8, 20)
    assert torch.equal(reference, first) and torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not restored.training and not any(p.requires_grad for p in restored.parameters())
    # No initializer or training dataset files were provided.
    with pytest.raises(ValueError, match="boundary pose|initial pose"):
        restored(tokens, None if task == "tracking" else torch.ones(1, 20))


@pytest.mark.parametrize("name", ["config.json", "model.safetensors", "codebooks.safetensors"])
def test_byte_integrity(bundle_factory, name):
    root, _ = bundle_factory()
    with (root / name).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="byte-size"):
        load_pose_model_bundle(root)


@pytest.mark.parametrize("fault", ["missing", "dtype", "shape", "nonfinite"])
def test_strict_tensor_loading_even_with_updated_manifest(bundle_factory, fault):
    root, _ = bundle_factory()
    tensors = load_file(root / "model.safetensors")
    key = "backbone.ln_f.weight"
    if fault == "missing":
        tensors.pop(key)
    elif fault == "dtype":
        tensors[key] = tensors[key].double()
    elif fault == "shape":
        tensors[key] = tensors[key][:-1]
    else:
        tensors[key][0] = float("nan")
    save_file(tensors, root / "model.safetensors")
    update_record(root, "model.safetensors", tensors)
    with pytest.raises(ValueError, match="incompatible|non-finite"):
        load_pose_model_bundle(root)


@pytest.mark.parametrize("fault", ["task", "offset", "lineage", "frontend"])
def test_contract_conflicts(bundle_factory, fault):
    root, _ = bundle_factory()
    document = json.loads((root / "config.json").read_text())
    if fault == "task":
        document["task"] = "tracking"
    elif fault == "offset":
        document["evaluation_contract"]["pose_offset_frames"] = 1
    elif fault == "lineage":
        document["initial_model"]["sha256"] = "f" * 64
    else:
        document["backbone_model_config"]["frame_pool"] = "mean"
    (root / "config.json").write_text(json.dumps(document))
    update_record(root, "config.json")
    with pytest.raises(ValueError):
        load_pose_model_bundle(root)


def test_raw_api_rejects_incompatible_frontend_before_tokenizer(bundle_factory, tmp_path):
    root, _ = bundle_factory()
    with pytest.raises(ValueError, match="frontend"):
        PosePredictor(root, tmp_path / "absent.pt")


def test_no_random_model_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        PosePredictor(tmp_path, tmp_path / "absent.pt")


def test_invalid_token_ids(bundle_factory):
    _, model = bundle_factory()
    with pytest.raises(ValueError, match="vocabulary"):
        model(torch.full((1, 6, 2, 2, 2), 8, dtype=torch.int64))
