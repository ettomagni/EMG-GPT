# Third-party notices

The repository's code license is [CC BY-NC 4.0](LICENSE). The terms below also
apply to the corresponding retained third-party components.

## NeuroRVQ compatibility implementation

`src/emg_gpt/neurorvq/` is adapted from
[KonstantinosBarmpas/NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ/tree/926e770d9d16b6aa308404280fa0cc0211a6f9fb)
at public revision `926e770d9d16b6aa308404280fa0cc0211a6f9fb`.
Its CC BY-NC 4.0 terms are included in [LICENSE](LICENSE).

The implementation is modified for EMG-GPT's package imports and token API,
retaining checkpoint-compatible topology. Attribution does not imply endorsement.

## Earlier implementation lineage

The tokenizer's source headers identify LaBraM, BEiT-v2/UNILM, timm, DeiT and
DINO. The timm utility source references v0.4.12. Original notices are preserved;
the texts below are included in the source archive and installed package.

| Upstream | License snapshot revision | Included text |
| --- | --- | --- |
| 935963004/LaBraM | `c431221e6cfd23dbfa9950e0180682fb322b0548` | [labram license](licenses/labram-LICENSE.txt) |
| microsoft/unilm | `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` | [unilm license](licenses/unilm-LICENSE.txt) |
| huggingface/pytorch-image-models | `7096b52a613eefb4f6d8107366611c8983478b19` | [timm license](licenses/timm-LICENSE.txt) |
| facebookresearch/deit | `7e160fe43f0252d17191b71cbb5826254114ea5b` | [deit license](licenses/deit-LICENSE.txt) |
| facebookresearch/dino | `7c446df5b9f45747937fb0d72314eb9f7b66930a` | [dino license](licenses/dino-LICENSE.txt) |

These revisions identify the license snapshots, not necessarily the original
revision of each inherited code fragment.

## External code and assets

The pinned external [facebookresearch/emg2pose checkout](https://github.com/facebookresearch/emg2pose/tree/5f6f62b1a0a08426adffe55900842e75a8adb38c)
is revision `5f6f62b1a0a08426adffe55900842e75a8adb38c`, with its declared
CC BY-NC-SA 4.0 terms. Its UmeTrack hand assets have their own CC BY-NC 4.0
license. The README links to the rights holder's downloads and terms.

This repository does not redistribute emg2pose recordings, official source,
hand meshes, or external checkpoints. Its tokenizer download points to the
NeuroRVQ authors' pinned Hugging Face artifact, which retains its upstream terms.
