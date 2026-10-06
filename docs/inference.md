# Inference contract

All output times use the nominal 2-kHz sample-index grid, relative to the first
sample. Official HDF5 acquisition timestamps contain clock jitter; they are
checked for monotonicity but do not replace this historical alignment.
The supplied channels must correspond to emg2pose electrodes, not arbitrary
sensor names. Incoming columns are mapped explicitly to the tokenizer order:
`c1,c10,c11,c12,c13,c14,c15,c16,c2,c3,c4,c5,c6,c7,c8,c9`.

The frontend uses third-order Butterworth SOS filtering (`sosfiltfilt`) at
20–399.5 Hz and `resample_poly(1,2)`, with the historical float32 casts.
No additional normalization is applied. Each independent 200-ms patch uses the
last learned temporal embedding (index 255 of 256); the tokenizer produces four
branches and four RVQ levels at 25 Hz. Low-level pose input is int64 `[batch,150,16,4,4]`.
The 16-frame storage block does not determine the embedding index.

Historical storage rounded token counts down to multiples of 16. Window planning
preserves that truncation and the original full-context raw-sample bound. Short
recordings are rejected; partial contexts are not padded. `max_windows` limits
prediction only, not the recording supplied to the zero-phase filter.

| Setting | Regression | Tracking |
| --- | --- | --- |
| Context / scored frames | 150 / 125 | 150 / 125 |
| Window stride (token frames) | 150 | 125 |
| Pose offset (token frames) | 0 | 1 |
| First scored time | 1.20 s | 1.24 s |
| Initial pose | None | Explicit, at first scored time of each window |

For token start `s` and output index `j=0..249`, the raw sample index is
`400 + (s + 25 + offset) * 80 + j * 40`. Divide by 2000 for seconds.
Tracking repeats features to 50 Hz and adds its first predicted angular increment
to the supplied boundary pose immediately, including at the boundary timestamp.
This reproduces the historical decoder convention. It does not clamp that first
output to the supplied pose. Regression interpolates features and internally
initializes its rollout. Each window resets recurrent state.

## Tracking input

Plan without loading weights or targets:

```python
from emg_gpt import plan_windows

plan = plan_windows(n_samples=len(raw_emg), task="tracking")
times = plan.boundary_timestamps_s
```

Supply `initial_poses_rad` of shape `[len(times),20]` and
`boundary_timestamps_s=times` to `PosePredictor.predict`. These are caller-provided
valid measurements in radians; the API checks shape, finiteness and alignment.
All-zero boundary rows are rejected, matching emg2pose's invalid-IK convention.
Angles exceeding one full turn (`2*pi` radians) are also rejected as a unit sanity
check. This is not an anatomical validator and cannot detect every degrees/radians
mix-up. Invalid boundaries fail the request rather than silently dropping windows.
Use the same `max_windows` for planning and predicting if limiting a run.

For the CLI, prepare the measured poses in the joint order below and save:

```python
import numpy as np
from emg_gpt import plan_windows

plan = plan_windows(len(raw_emg), "tracking", max_windows=1)
# measured_poses_rad: one valid measurement per plan.boundary_timestamps_s.
np.savez("boundary.npz", initial_poses_rad=measured_poses_rad,
         boundary_timestamps_s=plan.boundary_timestamps_s)
```

Pass `--model-dir weights/tracking --boundary-poses boundary.npz --max-windows 1`
to the same prediction command used for Regression.
Never label an independently carried prediction as ground-truth initialization.

## Prediction NPZ

Load with `numpy.load(path, allow_pickle=False)`:

| Field | Meaning |
| --- | --- |
| `joint_angles_rad` | Float32 `[W,250,20]` angles |
| `timestamps_s`, `source_sample_indices` | `[W,250]` recording-relative times and 2-kHz indices |
| `joint_names` | 20 names, ordered as the last angle dimension |
| `window_token_starts` | `[W]` context starts at 25 Hz |
| `valid` | `[W,250]` finite-output flags; not dataset ground-truth validity |
| `coverage_mask`, `coverage_timestamps_s` | Full recording's 50-Hz grid, true only at predicted times |
| `metadata_json` | Scalar JSON: task, units, preprocessing, window policy, code version, device, token batch size/index, pinned-bundle status and artifact hashes |

Uncovered times have no predicted angle. There is no interpolation across gaps.
Joint order is thumb CMC-FE, CMC-AA, MCP-FE, IP-FE; then index, middle, ring and
pinky, each MCP-AA, MCP-FE, PIP-FE, DIP-FE. FE denotes flexion/extension and AA
abduction/adduction, using the pinned emg2pose convention. Output contains joint
angles only, not fingertip positions or hand meshes.

Saved predictions and downloaded tokenizer files default to owner-only access
(`0600`). Set group permissions explicitly if sharing these files.

## Reproducibility

The corrected release was compared with the original research tokenizer, dataset
and pose checkpoints on two validation recordings, two windows per task and
recording: 303,104 token-ID comparisons and 40,000 predicted joint angles matched
exactly. Both tasks also passed API/CLI output parity. This check used CPU,
Python 3.12, PyTorch 2.14.1 and token batch size 16; it is not a benchmark rerun.

`PosePredictor` and the CLI require an official bundle from the packaged artifact
catalog. The low-level `emg_gpt.bundle.load_pose_model_bundle` can load other
self-consistent bundles; use `require_pinned=True` to require official identity.
Integrity against a bundle's own manifest alone does not establish that identity.

Exact parity requires the same device, runtime, preprocessing and token batch
size (default 16). Small floating-point differences can change a nearest-code
decision; bitwise equivalence across batch sizes or hardware is not promised.
The pinned CPU constraints and minimum-runtime constraints serve different tests.
