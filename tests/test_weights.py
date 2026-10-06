"""Full raw-EMG inference against frozen outputs from the research implementation."""

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from reference_input import synthetic_boundary, synthetic_emg

import emg_gpt.inference as inference
from emg_gpt.artifacts import sha256
from emg_gpt.tokenization import RAW_CHANNELS, load_tokenizer, preprocess_emg, tokenize_frames

pytestmark = pytest.mark.weights
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def weights(request):
    directory = os.environ.get("EMG_GPT_WEIGHTS")
    if not directory:
        if "weights" in request.config.getoption("markexpr"):
            pytest.fail(
                "Set EMG_GPT_WEIGHTS to the directory containing both bundles and tokenizer"
            )
        pytest.skip("Set EMG_GPT_WEIGHTS to run checkpoint integration tests")
    root = Path(directory).resolve()
    for name in (
        "regression/model.safetensors",
        "tracking/model.safetensors",
        "NeuroRVQ_EMG_tokenizer_v1.pt",
    ):
        assert (root / name).is_file(), f"Missing weight file: {root / name}"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield root
    torch.set_num_threads(previous_threads)


@pytest.fixture(scope="module")
def reference(weights):
    metadata = json.loads((FIXTURES / "reference.json").read_text())
    assert sha256(FIXTURES / "reference.npz") == metadata["reference_sha256"]
    assert (
        hashlib.sha256(synthetic_emg().astype("<f4").tobytes()).hexdigest()
        == metadata["input_sha256"]
    )
    with np.load(FIXTURES / "reference.npz", allow_pickle=False) as data:
        yield data


@pytest.mark.parametrize("task", ["regression", "tracking"])
def test_original_raw_to_pose(weights, reference, monkeypatch, task):
    predictor = inference.PosePredictor(
        weights / task, weights / "NeuroRVQ_EMG_tokenizer_v1.pt", device="cpu"
    )
    captured = []

    def capture(*args, **kwargs):
        tokens = tokenize_frames(*args, **kwargs)
        captured.append(tokens)
        return tokens

    monkeypatch.setattr(inference, "tokenize_frames", capture)
    boundary = (
        {}
        if task == "regression"
        else {"initial_poses_rad": synthetic_boundary(), "boundary_timestamps_s": np.array([1.24])}
    )
    result = predictor.predict(
        synthetic_emg(), sampling_rate_hz=2000, channel_names=RAW_CHANNELS, **boundary
    )
    assert len(captured) == 1
    # Nearby RVQ codes can switch under different CPU kernels; pose agreement is also required.
    assert np.mean(captured[0] == reference["tokens"]) >= 0.9999
    np.testing.assert_allclose(result.joint_angles_rad, reference[task], rtol=0, atol=1e-5)
    start = 2400 if task == "regression" else 2480
    np.testing.assert_array_equal(
        result.source_sample_indices, np.arange(start, start + 10000, 40)[None]
    )
    assert result.coverage_mask.sum() == 250
    del predictor
    gc.collect()


def test_reference_detects_wrong_temporal_embedding(weights, reference):
    tokenizer = load_tokenizer(weights / "NeuroRVQ_EMG_tokenizer_v1.pt", torch.device("cpu"))
    # Reproduce the historical 15/255 error with real weights and a non-constant input.
    with torch.no_grad():
        tokenizer.encoder.time_embed[-1].copy_(tokenizer.encoder.time_embed[15])
    wrong = tokenize_frames(
        tokenizer, preprocess_emg(synthetic_emg()), 16, batch_size=16, device=torch.device("cpu")
    )
    assert np.mean(wrong == reference["tokens"][:16]) < 0.99
    del tokenizer
    gc.collect()
