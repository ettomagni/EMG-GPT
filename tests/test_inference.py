import json

import numpy as np
import pytest
import torch

import emg_gpt.inference as inference
from emg_gpt import PosePredictor, plan_windows
from emg_gpt.tokenization import RAW_CHANNELS


@pytest.mark.parametrize("task,first,stride", [("regression", 1.20, 150), ("tracking", 1.24, 125)])
def test_window_alignment(task, first, stride):
    plan = plan_windows(40000, task)
    assert plan.timestamps_s[0, 0] == first
    assert np.all(np.diff(plan.token_starts) == stride)
    assert np.all(np.diff(plan.source_sample_indices, axis=1) == 40)
    assert np.all(plan.source_sample_indices < 40000)
    assert plan.stored_token_frames % 16 == 0
    assert plan.source_sample_indices.shape[1] == 250
    limited = plan_windows(40000, task, max_windows=1)
    assert len(limited.token_starts) == 1 and limited.total_windows == len(plan.token_starts)


@pytest.mark.parametrize("task", ["regression", "tracking"])
def test_short_context_and_storage_boundary(task):
    with pytest.raises(ValueError, match="too short"):
        plan_windows(13118, task)
    assert len(plan_windows(13119, task).token_starts) == 1


@pytest.mark.parametrize(
    "n,task,limit",
    [
        (True, "regression", None),
        (0, "regression", None),
        (14000, "other", None),
        (14000, "tracking", 0),
        (14000, "regression", True),
    ],
)
def test_invalid_plan(n, task, limit):
    with pytest.raises(ValueError):
        plan_windows(n, task, max_windows=limit)


@pytest.fixture
def fake_predictor(monkeypatch):
    """Exercise API contracts only; real-weight parity is a separate manual check."""

    class Decoder:
        def eval(self):
            return self

        def __call__(self, tokens, initial_pose=None):
            output = torch.zeros((1, 250, 20))
            return output if initial_pose is None else output + initial_pose[:, None, :]

    def make(task="regression"):
        obj = PosePredictor.__new__(PosePredictor)
        obj.device, obj.task, obj.token_batch_size = torch.device("cpu"), task, 16
        obj.tokenizer, obj.model, obj.identity = object(), Decoder(), {"fixture": "synthetic"}
        monkeypatch.setattr(
            inference,
            "tokenize_frames",
            lambda tok, emg, n, **kw: np.zeros((n, 16, 4, 4), dtype=np.uint16),
        )
        return obj

    return make


@pytest.mark.parametrize(
    "fault", ["rate", "channels", "shape", "nan", "complex", "short", "unexpected_pose"]
)
def test_raw_input_errors(fake_predictor, fault):
    model = fake_predictor()
    raw = np.zeros((14000, 16), dtype=np.float32)
    options = {"sampling_rate_hz": 2000, "channel_names": RAW_CHANNELS}
    if fault == "rate":
        options["sampling_rate_hz"] = 1000
    elif fault == "channels":
        options["channel_names"] = ["c1"] * 16
    elif fault == "shape":
        raw = raw.T
    elif fault == "nan":
        raw[0, 0] = np.nan
    elif fault == "complex":
        raw = raw.astype(complex)
    elif fault == "short":
        raw = raw[:100]
    else:
        options["initial_poses_rad"] = np.zeros((1, 20))
    with pytest.raises(ValueError):
        model.predict(raw, **options)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "shape",
        "nan",
        "infinity",
        "nan_and_infinity",
        "timestamp",
        "zeros",
        "degrees",
        "negative_degrees",
        "integer_limit",
    ],
)
def test_tracking_requires_aligned_explicit_boundaries(fake_predictor, fault):
    model = fake_predictor("tracking")
    options = {}
    if fault != "missing":
        options = {"initial_poses_rad": np.ones((1, 20)), "boundary_timestamps_s": np.array([1.24])}
        if fault == "shape":
            options["initial_poses_rad"] = np.ones(20)
        elif fault == "nan":
            options["initial_poses_rad"][0, 0] = np.nan
        elif fault == "infinity":
            options["initial_poses_rad"][:] = np.inf
        elif fault == "nan_and_infinity":
            options["initial_poses_rad"][:] = np.nan
            options["initial_poses_rad"][0, 0] = np.inf
        elif fault == "timestamp":
            options["boundary_timestamps_s"][0] = 1.20
        elif fault == "zeros":
            options["initial_poses_rad"][:] = 0
        elif fault == "integer_limit":
            options["initial_poses_rad"] = np.full((1, 20), np.iinfo(np.int64).min)
        else:
            options["initial_poses_rad"][0, 0] = 45 if fault == "degrees" else -45
    with pytest.raises(ValueError):
        model.predict(
            np.zeros((14000, 16)), sampling_rate_hz=2000, channel_names=RAW_CHANNELS, **options
        )


def test_tracking_allows_zero_joints_in_a_valid_pose(fake_predictor):
    model = fake_predictor("tracking")
    initial = np.zeros((1, 20), dtype=np.float32)
    initial[0, 1] = 0.2
    prediction = model.predict(
        np.zeros((14000, 16)),
        sampling_rate_hz=2000,
        channel_names=RAW_CHANNELS,
        initial_poses_rad=initial,
        boundary_timestamps_s=np.array([1.24]),
    )
    assert np.array_equal(prediction.joint_angles_rad[0, 0], initial[0])


@pytest.mark.parametrize("missing", [(0,), (1,), (2,), (0, 1, 2)])
def test_tracking_skips_missing_boundaries_without_shifting_windows(
    fake_predictor, monkeypatch, tmp_path, missing
):
    model = fake_predictor("tracking")
    raw = np.zeros((40000, 16), dtype=np.float32)
    plan = model.plan(len(raw))
    initial = np.repeat(np.array([0.1, 0.2, 0.3])[:, None], 20, axis=1)
    expected = model.predict(
        raw,
        sampling_rate_hz=2000,
        channel_names=RAW_CHANNELS,
        initial_poses_rad=initial,
        boundary_timestamps_s=plan.boundary_timestamps_s,
    )
    observed = []
    token_counts = []
    decoder = model.model
    tokenize = inference.tokenize_frames

    class RecordingDecoder:
        def eval(self):
            return self

        def __call__(self, tokens, initial_pose=None):
            assert torch.isfinite(initial_pose).all()
            observed.append(initial_pose.cpu().numpy().copy())
            return decoder(tokens, initial_pose=initial_pose)

    def capture(tokenizer, processed, count, **kwargs):
        assert processed.shape == (16, len(raw) // 2)
        token_counts.append(count)
        return tokenize(tokenizer, processed, count, **kwargs)

    model.model = RecordingDecoder()
    monkeypatch.setattr(inference, "tokenize_frames", capture)
    initial[list(missing)] = np.nan
    pred = model.predict(
        raw,
        sampling_rate_hz=2000,
        channel_names=RAW_CHANNELS,
        initial_poses_rad=initial,
        boundary_timestamps_s=plan.boundary_timestamps_s,
    )
    valid_windows = np.isfinite(initial).all(axis=1)
    assert len(observed) == int(valid_windows.sum())
    assert token_counts == [((int(plan.token_starts[-1]) + 150 + 15) // 16) * 16]
    assert np.array_equal(
        pred.joint_angles_rad[valid_windows], expected.joint_angles_rad[valid_windows]
    )
    assert np.isnan(pred.joint_angles_rad[~valid_windows]).all()
    assert np.array_equal(pred.source_sample_indices, plan.source_sample_indices)
    assert np.array_equal(pred.window_token_starts, plan.token_starts)
    assert pred.coverage_mask.sum() == valid_windows.sum() * 250
    assert not pred.coverage_mask[plan.source_sample_indices[~valid_windows] // 40].any()
    assert pred.metadata["schema_version"] == 2
    assert pred.metadata["planned_windows"] == 3
    assert pred.metadata["predicted_windows"] == int(valid_windows.sum())
    assert pred.metadata["skipped_windows"] == len(missing)
    assert pred.metadata["skipped_window_indices"] == list(missing)
    assert pred.metadata["skipped_window_reason"] == "missing_boundary_pose"
    destination = tmp_path / "skipped.npz"
    pred.save(destination)
    with np.load(destination, allow_pickle=False) as saved:
        assert np.array_equal(saved["valid"], np.repeat(valid_windows[:, None], 250, axis=1))
        assert np.array_equal(saved["coverage_mask"], pred.coverage_mask)


def test_missing_boundary_still_requires_aligned_timestamp(fake_predictor):
    with pytest.raises(ValueError, match="timestamps"):
        fake_predictor("tracking").predict(
            np.zeros((14000, 16)),
            sampling_rate_hz=2000,
            channel_names=RAW_CHANNELS,
            initial_poses_rad=np.full((1, 20), np.nan),
            boundary_timestamps_s=np.array([1.2]),
        )


def test_max_windows_counts_skipped_slots(fake_predictor):
    model = fake_predictor("tracking")
    raw = np.zeros((40000, 16))
    pred = model.predict(
        raw,
        sampling_rate_hz=2000,
        channel_names=RAW_CHANNELS,
        max_windows=1,
        initial_poses_rad=np.full((1, 20), np.nan),
        boundary_timestamps_s=model.plan(len(raw), max_windows=1).boundary_timestamps_s,
    )
    assert pred.joint_angles_rad.shape == (1, 250, 20)
    assert pred.metadata["total_available_windows"] == 3
    assert pred.metadata["predicted_windows"] == 0 and not pred.coverage_mask.any()


def test_cli_tracking_preserves_skipped_window(fake_predictor, monkeypatch, tmp_path, capsys):
    from emg_gpt import cli

    model = fake_predictor("tracking")
    monkeypatch.setattr(cli, "PosePredictor", lambda *args, **kwargs: model)
    raw = np.zeros((40000, 16), dtype=np.float32)
    initial = np.ones((3, 20), dtype=np.float32)
    initial[1] = np.nan
    np.savez(
        tmp_path / "raw.npz", emg=raw, sampling_rate_hz=2000, channel_names=np.asarray(RAW_CHANNELS)
    )
    np.savez(
        tmp_path / "boundary.npz",
        initial_poses_rad=initial,
        boundary_timestamps_s=model.plan(len(raw)).boundary_timestamps_s,
    )
    cli.main(
        [
            "--model-dir",
            str(tmp_path),
            "--tokenizer",
            "unused.pt",
            "--input",
            str(tmp_path / "raw.npz"),
            "--output",
            str(tmp_path / "prediction.npz"),
            "--boundary-poses",
            str(tmp_path / "boundary.npz"),
        ]
    )
    with np.load(tmp_path / "prediction.npz", allow_pickle=False) as saved:
        assert saved["joint_angles_rad"].shape == (3, 250, 20)
        assert np.isnan(saved["joint_angles_rad"][1]).all()
        assert saved["valid"][[0, 2]].all() and not saved["valid"][1].any()
    output = capsys.readouterr().out
    assert "2 predicted tracking windows" in output and "1 skipped" in output


@pytest.mark.parametrize("task", ["regression", "tracking"])
def test_save_coverage_and_joint_order(fake_predictor, tmp_path, task):
    model = fake_predictor(task)
    raw = np.zeros((40000, 16), dtype=np.float32)
    plan = model.plan(len(raw))
    options = (
        {}
        if task == "regression"
        else {
            "initial_poses_rad": np.ones((len(plan.token_starts), 20)),
            "boundary_timestamps_s": plan.boundary_timestamps_s,
        }
    )
    pred = model.predict(raw, sampling_rate_hz=2000, channel_names=RAW_CHANNELS, **options)
    destination = tmp_path / "prediction.npz"
    pred.save(destination)
    with np.load(destination, allow_pickle=False) as data:
        assert data["joint_angles_rad"].shape == (len(plan.token_starts), 250, 20)
        assert data["joint_names"].tolist() == list(inference.JOINT_NAMES)
        assert data["valid"].all()
        assert data["coverage_mask"].sum() == len(plan.token_starts) * 250
        assert not data["coverage_mask"][0] and not data["coverage_mask"][-1]
        assert json.loads(data["metadata_json"].item())["task"] == task
        assert data["joint_angles_rad"][0, 0, 0] == int(task == "tracking")
    with pytest.raises(FileExistsError):
        pred.save(destination)


def test_channel_permutation_and_whole_record_filter(fake_predictor, monkeypatch):
    model = fake_predictor()
    observed = []
    original = inference.preprocess_emg

    def capture(raw):
        observed.append(raw.copy())
        return original(raw)

    monkeypatch.setattr(inference, "preprocess_emg", capture)
    raw = np.random.default_rng(3).normal(size=(40000, 16)).astype(np.float32)
    model.predict(
        raw[:, ::-1], sampling_rate_hz=2000, channel_names=RAW_CHANNELS[::-1], max_windows=1
    )
    assert np.array_equal(observed[0], raw)


def test_nonfinite_prediction_fails(fake_predictor):
    model = fake_predictor()

    class BrokenDecoder:
        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            return torch.full((1, 250, 20), float("nan"))

    model.model = BrokenDecoder()
    with pytest.raises(ValueError, match="invalid pose"):
        model.predict(np.zeros((14000, 16)), sampling_rate_hz=2000, channel_names=RAW_CHANNELS)
