import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from emg_gpt import cli
from emg_gpt.artifacts import fetch_tokenizer, resource_json, sha256, verify_file
from emg_gpt.inference import JOINT_NAMES, PosePrediction
from emg_gpt.tokenization import RAW_CHANNELS, load_tokenizer


def test_packaged_resources():
    assert resource_json("tokenizer.json")["n_code"] == 8192
    assert set(resource_json("artifacts.json")["pose_models"]) == {"regression", "tracking"}


def test_integrity_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"not a checkpoint")

    def forbidden(*args, **kwargs):
        pytest.fail("Must verify identity before deserializing")

    monkeypatch.setattr("torch.load", forbidden)
    with pytest.raises(ValueError, match="SHA-256"):
        load_tokenizer(path, "cpu")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_tokenizer(path)
    verify_file(path, {"bytes": path.stat().st_size, "sha256": sha256(path)})


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission semantics")
@pytest.mark.parametrize("mask", [0o022, 0o002, 0o077])
def test_saved_predictions_and_tokenizer_follow_umask(tmp_path, monkeypatch, mask):
    cached = tmp_path / "cached.pt"
    cached.write_bytes(b"tokenizer download fixture")
    cached.chmod(0o600)
    artifact = {
        "bytes": cached.stat().st_size,
        "sha256": sha256(cached),
        "public_source": {"repo_id": "fixture", "revision": "a" * 40, "filename": cached.name},
    }
    monkeypatch.setattr("emg_gpt.artifacts.resource_json", lambda _: {"tokenizer": artifact})
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kwargs: str(cached)),
    )
    prediction = PosePrediction(
        np.zeros((1, 250, 20), dtype=np.float32),
        np.arange(2400, 12400, 40)[None],
        np.zeros(350, dtype=bool),
        np.array([0]),
        {"synthetic": True},
    )
    output = tmp_path / "output"
    previous = os.umask(mask)
    try:
        prediction.save(output / "prediction.npz")
        fetch_tokenizer(output / "tokenizer.pt")
    finally:
        os.umask(previous)
    assert {path.name for path in output.iterdir()} == {"prediction.npz", "tokenizer.pt"}
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~mask
    assert stat.S_IMODE(cached.stat().st_mode) == 0o600
    assert (output / "tokenizer.pt").read_bytes() == cached.read_bytes()
    with np.load(output / "prediction.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["joint_angles_rad"], prediction.joint_angles_rad)


def test_npz_input_metadata(tmp_path):
    path = tmp_path / "raw.npz"
    np.savez(
        path,
        emg=np.zeros((14000, 16)),
        sampling_rate_hz=2000,
        channel_names=np.asarray(RAW_CHANNELS),
    )
    raw, rate, channels = cli.read_emg(path)
    assert raw.shape == (14000, 16) and rate == 2000 and channels == RAW_CHANNELS
    np.savez(path, emg=raw)
    with pytest.raises(ValueError, match="lacks"):
        cli.read_emg(path)
    np.savez(
        path, emg=raw, sampling_rate_hz=2000, channel_names=np.asarray(RAW_CHANNELS, dtype=object)
    )
    with pytest.raises(ValueError, match="Object arrays"):
        cli.read_emg(path)


def test_hdf5_does_not_require_or_read_pose(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "recording.hdf5"
    data = np.zeros(14000, dtype=[("emg", "f4", (16,)), ("time", "f8")])
    data["time"] = 100 + np.arange(len(data)) / 2000
    with h5py.File(path, "w") as f:
        group = f.create_group("emg2pose")
        group.attrs["sample_rate"] = 2000
        group.create_dataset("timeseries", data=data)
    raw, rate, channels = cli.read_emg(path)
    assert raw.shape == (14000, 16) and rate == 2000 and channels == RAW_CHANNELS
    with h5py.File(path, "r+") as f:
        f["emg2pose"].attrs["sample_rate"] = 1000
    with pytest.raises(ValueError, match="sample_rate"):
        cli.read_emg(path)


def test_cli_delegates_to_api_and_preserves_output(tmp_path, monkeypatch):
    raw = tmp_path / "raw.npz"
    out = tmp_path / "prediction.npz"
    np.savez(
        raw,
        emg=np.zeros((14000, 16)),
        sampling_rate_hz=2000,
        channel_names=np.asarray(RAW_CHANNELS),
    )
    calls = []

    class Predictor:
        task = "regression"

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def predict(self, raw, **kwargs):
            calls.append((raw, kwargs))
            return PosePrediction(
                np.zeros((1, 250, 20), dtype=np.float32),
                np.arange(2400, 12400, 40)[None],
                np.zeros(350, dtype=bool),
                np.array([0]),
                {"synthetic": True, "predicted_windows": 1, "skipped_windows": 0},
            )

    monkeypatch.setattr(cli, "PosePredictor", Predictor)
    args = [
        "--model-dir",
        str(tmp_path),
        "--tokenizer",
        str(tmp_path / "unused.pt"),
        "--input",
        str(raw),
        "--output",
        str(out),
        "--max-windows",
        "1",
    ]
    cli.main(args)
    assert len(calls) == 2 and calls[1][1]["max_windows"] == 1
    with np.load(out, allow_pickle=False) as prediction:
        assert prediction["joint_names"].tolist() == list(JOINT_NAMES)
        assert json.loads(prediction["metadata_json"].item())["synthetic"]
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2 and len(calls) == 2


def test_cli_help_and_runtime_import_isolation():
    result = subprocess.run(
        [sys.executable, "-m", "emg_gpt.cli", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0 and "--boundary-poses" in result.stdout
    code = "from emg_gpt import PosePredictor; import sys; assert not any(x in sys.modules for x in ['pandas','h5py','emg_gpt.training_state','emg_pose','NeuroRVQ_EMG','emg2pose'])"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_vendored_packages_do_not_import_names_from_working_directory(tmp_path):
    for name in ("NeuroRVQ_EMG", "emg_pose"):
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('unrelated package imported')")
    result = subprocess.run(
        [sys.executable, "-c", "from emg_gpt import PosePredictor"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", ["missing", "pickle", "not_found"])
def test_cli_boundary_errors_name_the_argument(tmp_path, capsys, fault):
    raw = tmp_path / "raw.npz"
    np.savez(
        raw,
        emg=np.zeros((14000, 16)),
        sampling_rate_hz=2000,
        channel_names=np.asarray(RAW_CHANNELS),
    )
    boundary = tmp_path / "boundary.npz"
    if fault == "missing":
        np.savez(boundary, initial_poses_rad=np.ones((1, 20)))
    elif fault == "pickle":
        np.savez(
            boundary,
            initial_poses_rad=np.ones((1, 20), dtype=object),
            boundary_timestamps_s=np.array([1.24]),
        )
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--model-dir",
                str(tmp_path),
                "--tokenizer",
                "unused.pt",
                "--input",
                str(raw),
                "--output",
                str(tmp_path / "out.npz"),
                "--boundary-poses",
                str(boundary),
            ]
        )
    assert error.value.code == 2
    assert "--boundary-poses" in capsys.readouterr().err
