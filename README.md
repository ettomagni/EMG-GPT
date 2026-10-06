<div align="center">

# EMG-GPT

**Predictive Pretraining on Residual-Quantized EMG Tokens for Hand Pose Estimation**

[Paper](https://arxiv.org/abs/2610.05235) · [Weights (private)](https://huggingface.co/ettoremagni/EMG-GPT) · [Citation](#citation)

Ettore Magni · Rolandos Alexandros Potamias · Stefanos Zafeiriou · Konstantinos Barmpas

</div>

![EMG-GPT architecture: frozen NeuroRVQ tokenizer, frame encoder, causal GPT, and Regression or Tracking pose decoder](images/architecture.png)

Offline hand-pose inference from 16-channel EMG using a frozen
[NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ) tokenizer, an adapted
causal GPT and a state-conditioned pose decoder. Inference follows the figure's
two pose paths with fixed weights; the pretraining head is bypassed.
This release contains inference code only.

## Paper results

EMG-GPT test results from [Tables 1 and 3](https://arxiv.org/html/2610.05235v1),
verified against the archived statistics for the checkpoints listed below.
Values are mean ± sample SD **across users**, not across training seeds; lower is better.

| Task | Held-out condition | Angular MAE (°) ↓ | Landmark error (mm) ↓ |
| --- | --- | ---: | ---: |
| Regression | User | 12.8 ± 1.1 | 16.3 ± 1.3 |
| Regression | Stage | 16.3 ± 1.6 | 22.0 ± 1.9 |
| Regression | User, Stage | 16.6 ± 1.3 | 22.7 ± 1.4 |
| Tracking | User | 8.9 ± 0.9 | 11.6 ± 1.2 |
| Tracking | Stage | 12.4 ± 1.4 | 16.9 ± 1.8 |
| Tracking | User, Stage | 12.2 ± 1.3 | 16.9 ± 1.4 |

Tracking uses a ground-truth boundary pose; Regression does not.
These are the paper's archived evaluations, not new smoke-test scores.
The inference API returns joint angles; landmark evaluation and training are outside this release.

## Comparison on the same test samples

Appendix E.2, [Table 4](https://arxiv.org/html/2610.05235v1#A5.T4), compares released
vEMG2Pose weights with EMG-GPT at the same ordered test target indices within
each task, using the same valid-value counts, metrics and equal-user aggregation.
Values are mean ± sample SD across users.

| Task | Held-out condition | Model | Angular MAE (°) ↓ | Landmark error (mm) ↓ |
| --- | --- | --- | ---: | ---: |
| Regression | User | vEMG2Pose | 12.80 ± 1.31 | 16.29 ± 1.85 |
| Regression | User | EMG-GPT | 12.80 ± 1.08 | 16.30 ± 1.29 |
| Regression | Stage | vEMG2Pose | 15.40 ± 1.59 | 20.43 ± 2.09 |
| Regression | Stage | EMG-GPT | 16.30 ± 1.57 | 22.02 ± 1.92 |
| Regression | User, Stage | vEMG2Pose | 15.73 ± 1.38 | 21.25 ± 1.79 |
| Regression | User, Stage | EMG-GPT | 16.64 ± 1.30 | 22.70 ± 1.39 |
| Tracking | User | vEMG2Pose | 8.57 ± 0.95 | 11.11 ± 1.35 |
| Tracking | User | EMG-GPT | 8.89 ± 0.93 | 11.59 ± 1.24 |
| Tracking | Stage | vEMG2Pose | 11.50 ± 1.41 | 15.45 ± 1.81 |
| Tracking | Stage | EMG-GPT | 12.42 ± 1.43 | 16.88 ± 1.80 |
| Tracking | User, Stage | vEMG2Pose | 11.33 ± 0.97 | 15.64 ± 1.33 |
| Tracking | User, Stage | EMG-GPT | 12.18 ± 1.25 | 16.93 ± 1.36 |

Each model retains its native frontend. This local comparison does not reproduce
the original vEMG2Pose benchmark protocol or establish statistical superiority.
The corresponding evaluator is outside this inference-only package.

## Availability

**Complete pose bundles are hosted privately on Hugging Face. Public downloads are pending.**

| Artifact | Selected checkpoint | Availability |
| --- | --- | --- |
| NeuroRVQ tokenizer | Upstream EMG v1 | Pinned, verified download |
| Regression | GPT initialization 280k; downstream step 2,000 | Verified export; Hugging Face access required |
| Tracking | GPT initialization 400k; downstream step 5,000 | Verified export; Hugging Face access required |

Each pose directory must contain `config.json`, `model.safetensors`,
`codebooks.safetensors` and `manifest.json`. These bundles contain the **adapted
backbone and pose head together**. Original GPT initializers, warm-start weights
and training data are not needed. Recorded identities are in
[src/emg_gpt/resources/artifacts.json](src/emg_gpt/resources/artifacts.json).
Loading checks hashes, tensor keys, shapes, dtypes, protocol and tokenizer/codebook
compatibility. There is no fallback model.

## Install

Using Python 3.11 or 3.12 (repository access is required while the code is private):

```bash
git clone https://github.com/ettomagni/EMG-GPT.git
cd EMG-GPT
python -m venv .venv
source .venv/bin/activate
python -m pip install . -c constraints/cpu-tested.txt
```

Local inference needs PyTorch, NumPy, SciPy, einops and safetensors. Install
`'.[data]'` for official emg2pose HDF5 input or `'.[download]'` for the tokenizer
helper. CPU is the verified baseline. CUDA is selectable but has not been tested
for this refactor; MPS is not supported. No external hand model or meshes are needed.

## Regression quick start

With access to the [private model repository](https://huggingface.co/ettoremagni/EMG-GPT),
download the pinned Regression bundle into `weights/regression/`:

```bash
python -m pip install '.[download]' -c constraints/cpu-tested.txt
hf auth login
hf download ettoremagni/EMG-GPT \
  --revision 5b1b58089a780965dba0e35dfb97bd464d8b3abd \
  --include "regression/*" --local-dir weights
emg-gpt-download-tokenizer --output weights/NeuroRVQ_EMG_tokenizer_v1.pt
emg-gpt-predict \
  --model-dir weights/regression \
  --tokenizer weights/NeuroRVQ_EMG_tokenizer_v1.pt \
  --input recording.npz --output prediction.npz --device cpu
```

For Tracking, download `tracking/*` instead and supply the explicit boundary
poses described below. Both bundles use the same pinned Hugging Face revision.

Input NPZ files must contain `emg` (finite real array `[samples,16]`),
`sampling_rate_hz` (scalar `2000`) and `channel_names` (Unicode strings `c1` through
`c16`, in the supplied array's order). Use `numpy.savez` with these keys; object
arrays are rejected. Preserve the native emg2pose signal amplitude scale: the
source does not specify a physical voltage unit, and this API does not normalize
or convert it. Other sensors/scales are not validated.

Official emg2pose HDF5 recordings can be passed directly with the `data` extra.
Only EMG and timestamps are read, never target poses. Datasets are not bundled;
see the [emg2pose source and data terms](https://github.com/facebookresearch/emg2pose).
`--max-windows 1` limits decoding for a smoke check while preserving filtering
of the entire recording. Output files are never overwritten.

To check file handling before using real data, create a **synthetic smoke input**
and use `--input smoke-input.npz`. This constant signal does not test accuracy:

```bash
python - <<'PY'
import numpy as np
np.savez("smoke-input.npz", emg=np.zeros((14000, 16), dtype=np.float32),
         sampling_rate_hz=2000,
         channel_names=np.array([f"c{i}" for i in range(1, 17)]))
PY
```

The same path is available in Python:

```python
import numpy as np
from emg_gpt import PosePredictor

model = PosePredictor("weights/regression", "weights/NeuroRVQ_EMG_tokenizer_v1.pt")
with np.load("recording.npz", allow_pickle=False) as data:
    result = model.predict(
        data["emg"], sampling_rate_hz=data["sampling_rate_hz"].item(),
        channel_names=data["channel_names"].tolist(),
    )
result.save("prediction-python.npz")
```

## Protocol and output

The frontend filters the whole 2-kHz recording at 20–399.5 Hz, resamples to
1 kHz and tokenizes independent 200-sample patches every 40 samples. Zero-phase
filtering makes this an **offline** pipeline. Do not prefilter, resample, normalize
or process independent chunks and expect identical results.

Predictions have shape `[windows,250,20]`, in radians at 50 Hz. Each window uses
150 token frames and scores the final 125. A full window needs at least 13,119
raw samples. Regression starts at 1.20 s and leaves one-second gaps between scored
windows. Timestamps and a coverage mask expose warm-up, gaps and the unscored tail.
Joint order, saved fields and Tracking initialization are specified in
[the inference contract](docs/inference.md).

Tracking requires one explicit 20-joint boundary pose **per window**, at the
planner's boundary timestamps. The first boundary is 1.24 s. Recurrent state resets
between windows; ground truth is never inferred or extracted automatically.
Comparisons must use the same alignment, coverage and initialization protocol.

## Verification

```bash
python -m pip install '.[dev,data]' -c constraints/cpu-tested.txt
python -m pytest
ruff check src scripts tests
python scripts/check_release.py --profile review
python -m build
```

The 46 focused tests pass on CPU with Python 3.11 and 3.12. Separate real-checkpoint
checks for Regression and Tracking each match the original preprocessing, 40,960
token IDs and 250 predicted poses exactly on a fixed validation window.
These checks establish implementation compatibility, not new benchmark scores.

## Citation

```bibtex
@misc{magni2026emggpt,
  title={EMG-GPT: Predictive Pretraining on Residual-Quantized EMG Tokens for Hand Pose Estimation},
  author={Ettore Magni and Rolandos Alexandros Potamias and Stefanos Zafeiriou and Konstantinos Barmpas},
  year={2026},
  eprint={2610.05235},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.05235}
}
```

Machine-readable metadata: [CITATION.cff](CITATION.cff).
The frozen tokenizer is adapted from [NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ).
The inherited [CC BY-NC 4.0 license](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md) are retained. Maintainer confirmation
of the new-code license and weight redistribution remains pending.
