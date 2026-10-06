# Inference guide

## Input and Python API

Use raw 2-kHz emg2pose EMG in its native amplitude scale. NPZ files contain
`emg` (finite real `[samples,16]`), scalar `sampling_rate_hz=2000` and Unicode
`channel_names` (`c1` through `c16` in column order). Object arrays are rejected.
The `data` extra enables official HDF5 input; only EMG and timestamps are read.

```python
import numpy as np
from emg_gpt import PosePredictor

model = PosePredictor(
    "weights/regression", "weights/NeuroRVQ_EMG_tokenizer_v1.pt", device="cpu",
)
with np.load("recording.npz", allow_pickle=False) as data:
    result = model.predict(
        data["emg"], sampling_rate_hz=data["sampling_rate_hz"].item(),
        channel_names=data["channel_names"].tolist(),
    )
result.save("prediction-python.npz")
```

Use `device="cuda"` (CLI: `--device cuda`) for NVIDIA GPU inference with a
CUDA-enabled PyTorch installation. MPS is unsupported. Outputs are never overwritten.

## Smoke check

After installing and downloading the Regression weights as in the README:

```bash
python - <<'PYSMOKE'
import numpy as np
np.savez("smoke-input.npz", emg=np.zeros((14000, 16), dtype=np.float32),
         sampling_rate_hz=2000,
         channel_names=np.array([f"c{i}" for i in range(1, 17)]))
PYSMOKE
emg-gpt-predict --model-dir weights/regression \
  --tokenizer weights/NeuroRVQ_EMG_tokenizer_v1.pt \
  --input smoke-input.npz --output smoke-prediction.npz --device cpu
```

This synthetic signal tests file handling and execution, not model accuracy.
Use a new output filename when rerunning: predictions are never overwritten.

## Preprocessing and alignment

Supply the complete recording: third-order Butterworth filtering (`sosfiltfilt`,
20–399.5 Hz) followed by `resample_poly(1,2)` makes inference offline. The original
float32 casts are preserved; amplitudes are not normalized. Prefiltering or splitting
a recording into chunks changes predictions. Other sensors/scales are unvalidated.

Channels are reordered to `c1,c10,c11,c12,c13,c14,c15,c16,c2,c3,c4,c5,c6,c7,c8,c9`.
Independent 200-ms patches every 40 ms use temporal embedding 255. Tokens have four
branches and four RVQ levels; low-level pose input is int64 `[batch,150,16,4,4]`.

Planning preserves the original 16-frame storage truncation. A full window needs
13,119 raw samples; partial windows are not padded. `max_windows` limits decoding,
while filtering still uses the whole recording. Timestamps use the nominal 2-kHz
sample grid relative to the recording start; HDF5 clock jitter does not change it.

| Setting | Regression | Tracking |
| --- | --- | --- |
| Context / scored frames | 150 / 125 | 150 / 125 |
| Window stride (token frames) | 150 | 125 |
| Pose offset (token frames) | 0 | 1 |
| First scored time | 1.20 s | 1.24 s |
| Initial pose | None | Explicit, at first scored time of each window |

Regression leaves one-second gaps between scored windows. Coverage metadata exposes
warm-up, gaps and the unscored tail; comparisons must use the same coverage.

For token start `s` and output index `j=0..249`, the raw sample index is
`400 + (s + 25 + offset) * 80 + j * 40`. Divide by 2000 for seconds.
Tracking repeats features to 50 Hz and adds the first predicted increment at the
boundary timestamp. Regression interpolates features and initializes its own
rollout. Both tasks reset recurrent state for each window.

## Tracking input

Download the Tracking bundle:

```bash
hf download ettoremagni/EMG-GPT \
  --revision 812d159b4e7a4fb1c95da865f4f1e2635fa6522f \
  --include "tracking/*" --include "LICENSE" --local-dir weights
```

Plan the required measurement times without loading weights or targets:

```python
import numpy as np
from emg_gpt import plan_windows

with np.load("recording.npz", allow_pickle=False) as data:
    plan = plan_windows(n_samples=len(data["emg"]), task="tracking")
times = plan.boundary_timestamps_s
print(times)
```

Pass measurements as `initial_poses_rad` (`[len(times),20]`, radians) and
`boundary_timestamps_s=times` to the API. An all-NaN row skips that window:
its angles are NaN, `valid=False`, and coverage is false; timestamps are preserved.
Partial-NaN, infinite, all-zero or outside-±2π rows are rejected. Use the same
`max_windows` for planning and predicting; it includes skipped windows.

For the CLI, save your measurements at those times in `measured-poses.npy`,
shape `[len(times),20]`, in radians and the joint order below. Then run:

```python
import numpy as np
from emg_gpt import plan_windows

with np.load("recording.npz", allow_pickle=False) as data:
    plan = plan_windows(len(data["emg"]), "tracking")
initial = np.load("measured-poses.npy", allow_pickle=False).astype(np.float32)
missing = ~np.isfinite(initial).all(axis=1) | np.isclose(initial, 0).all(axis=1)
initial[missing] = np.nan
np.savez("boundary.npz", initial_poses_rad=initial,
         boundary_timestamps_s=plan.boundary_timestamps_s)
```

```bash
emg-gpt-predict --model-dir weights/tracking \
  --tokenizer weights/NeuroRVQ_EMG_tokenizer_v1.pt \
  --input recording.npz --boundary-poses boundary.npz \
  --output tracking-prediction.npz --device cpu
```

## Prediction NPZ

Load with `numpy.load(path, allow_pickle=False)`:

| Field | Meaning |
| --- | --- |
| `joint_angles_rad` | Float32 `[W,250,20]` angles; skipped windows contain NaN |
| `timestamps_s`, `source_sample_indices` | `[W,250]` recording-relative times and 2-kHz indices |
| `joint_names` | 20 names, ordered as the last angle dimension |
| `window_token_starts` | `[W]` context starts at 25 Hz |
| `valid` | `[W,250]` finite-output flags; not dataset ground-truth validity |
| `coverage_mask`, `coverage_timestamps_s` | Full recording's 50-Hz grid, true only at predicted times |
| `metadata_json` | Scalar JSON: task, units, preprocessing, window policy, code version, device, token batch size/index, pinned-bundle status and artifact hashes |

Uncovered times have no predicted angle. There is no interpolation across gaps.
Metadata schema 2 records `planned_windows`, `predicted_windows`, `skipped_windows`,
`skipped_window_indices` (zero-based output rows) and `skipped_window_reason`.
If every boundary is missing, the result keeps all planned rows with zero coverage.
Joint order is thumb CMC-FE, CMC-AA, MCP-FE, IP-FE; then index, middle, ring and
pinky, each MCP-AA, MCP-FE, PIP-FE, DIP-FE. FE denotes flexion/extension and AA
abduction/adduction, using the pinned emg2pose convention.

## Reproducibility

Install the tested CPU dependencies, then run the lightweight checks:

```bash
python -m pip install '.[dev,data,download]' -c constraints/cpu-tested.txt
python -m pytest -m "not weights"
ruff check src scripts tests
python scripts/check_release.py
cffconvert --validate
python -m build
```

After downloading **both** bundles and the tokenizer into `weights/`, run the
integration tests against frozen outputs from the original research implementation:

```bash
EMG_GPT_WEIGHTS="$PWD/weights" python -m pytest -m weights
```

The synthetic reference covers raw EMG through both decoders and detects tokenizer
misalignment. Its source/checkpoint hashes are recorded alongside it.
[`make_reference.py`](../tests/make_reference.py) requires the original research
environment and checkpoints to regenerate it; this is not an accuracy benchmark.

CI tests the installed wheel on Python 3.11–3.14 and minimum dependencies on 3.11.
It builds the wheel from the source archive and runs the weight tests on pushes
and manual runs. Release parity has also been checked on real recordings on CPU;
CUDA numerical parity remains unverified. Training used CUDA.

Use the same runtime, device and token batch size (default 16) for numerical
comparisons: floating-point differences can change RVQ code selection. The weight
tests require at least 99.99% token agreement and pose differences below `1e-5` rad.
`PosePredictor` and the CLI require official bundles with verified catalog hashes.
The low-level `load_pose_model_bundle` also supports self-consistent custom bundles;
pass `require_pinned=True` to require an official bundle.
