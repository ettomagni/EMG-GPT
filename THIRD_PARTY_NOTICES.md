# Third-party notices

License sign-off: **PENDING** for the new EMG-GPT code. The inherited license
and upstream notices below are included; maintainer confirmation of the
new-code license is still required.

## NeuroRVQ compatibility implementation

`src/NeuroRVQ_EMG/` is adapted from
[KonstantinosBarmpas/NeuroRVQ](https://github.com/KonstantinosBarmpas/NeuroRVQ/tree/926e770d9d16b6aa308404280fa0cc0211a6f9fb)
at public revision `926e770d9d16b6aa308404280fa0cc0211a6f9fb`.
Its [CC BY-NC 4.0 terms](LICENSE) are included; one malformed quotation in
the upstream text has normalized punctuation in this copy.

The implementation is modified for EMG-GPT's package imports and token API.
The encoder, decoder and quantizer topology is retained for strict loading of
the original tokenizer checkpoint. These adaptations are not an upstream
NeuroRVQ release, and attribution does not imply upstream endorsement.

## Earlier implementation lineage

The tokenizer's source headers identify LaBraM, BEiT-v2/UNILM, timm, DeiT and
DINO. The timm utility source explicitly references v0.4.12. The original
copyright and license texts below are distributed with both the source
archive and the installed package; their terms continue to apply to the
corresponding retained portions.

| Upstream | License snapshot revision | Included text |
| --- | --- | --- |
| 935963004/LaBraM | `c431221e6cfd23dbfa9950e0180682fb322b0548` | [labram license](licenses/labram-LICENSE.txt) |
| microsoft/unilm | `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` | [unilm license](licenses/unilm-LICENSE.txt) |
| huggingface/pytorch-image-models | `7096b52a613eefb4f6d8107366611c8983478b19` | [timm license](licenses/timm-LICENSE.txt) |
| facebookresearch/deit | `7e160fe43f0252d17191b71cbb5826254114ea5b` | [deit license](licenses/deit-LICENSE.txt) |
| facebookresearch/dino | `7c446df5b9f45747937fb0d72314eb9f7b66930a` | [dino license](licenses/dino-LICENSE.txt) |

These revisions identify the retrieved license texts. Except for the explicit
timm tag and NeuroRVQ base above, they do not establish which historical
revision each transitive fragment was originally copied from. Original lineage
comments are preserved. This inventory does not claim that all upstream code
is included or that one license replaces another.

## External code and assets

The pinned external [facebookresearch/emg2pose checkout](https://github.com/facebookresearch/emg2pose/tree/5f6f62b1a0a08426adffe55900842e75a8adb38c)
is revision `5f6f62b1a0a08426adffe55900842e75a8adb38c`, with its declared
CC BY-NC-SA 4.0 terms. Its UmeTrack hand assets have their own CC BY-NC 4.0
license. The README links to the rights holder's downloads and terms.

This repository does not redistribute emg2pose recordings, official source,
hand meshes, or external checkpoints. Its tokenizer download points to the
NeuroRVQ authors' pinned Hugging Face artifact. Exporting model weights or
other derived assets does not itself establish redistribution permission.
