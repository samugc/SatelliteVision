# SatelliteVision

[![CI](https://github.com/samugc/SatelliteVision/actions/workflows/ci.yml/badge.svg)](https://github.com/samugc/SatelliteVision/actions/workflows/ci.yml)

Satellite image classification on **xView** with a fine-tuned **Swin Transformer**, imbalance-aware training, test-time augmentation and explicit evaluation by image identity.

Originally developed as MSc Computer Vision coursework and later refactored for reproducibility and stricter evaluation.

## Highlights

- 13-class satellite crop classification.
- Full fine-tuning of `microsoft/swin-base-patch4-window7-224`.
- Class-weighted cross-entropy with MixUp/CutMix.
- Mixed-precision training and gradient accumulation.
- Four-view test-time augmentation (TTA).
- Prediction/label alignment by image ID and filename.
- Reproducible split manifests and lightweight integrity tests.

## Historical result

| Model | Split | Metric | Result |
|---|---|---:|---:|
| Swin-Base + TTA | validation | accuracy | **83.20%** |

The historical run used 16,871 training images and 1,875 validation images.

The 83.20% figure is a **validation result**, not a held-out test score. The original checkpoint and per-image predictions were not preserved, so the score is reported as historical evidence rather than as a result reproduced by the cleaned repository.

## Pipeline

```text
xView annotations
      ↓
ID-based train/validation split
      ↓
Swin-Base
      ↓
class weighting + MixUp/CutMix
      ↓
mixed precision + gradient accumulation
      ↓
validation + four-view TTA
      ↓
ID-aligned evaluation
```

The evaluator can report accuracy, fixed-13-class macro-F1, per-class precision/recall/F1, confusion matrix entries and misclassified examples for new prediction exports.

## Repository structure

```text
classification/     split and evaluation utilities
notebooks/          cleaned Swin training workflow
evidence/           historical outputs and split/provenance records
tests/              integrity and evaluation tests
VALIDATION.md       executed checks and known limits
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter lab notebooks/swin_classification.ipynb
```

The original xView images are not redistributed. Point `XVIEW_DATA` to the local dataset before running training cells.

New prediction exports can be evaluated without retraining:

```bash
python -m classification.core \
  evidence/validation_split.csv \
  runs/swin/predictions.json
```

## Reproducibility improvements

- persisted split identities;
- explicit image-ID/filename joins;
- rejection of missing or duplicated predictions;
- correct handling of the final gradient-accumulation group;
- structured metric/error exports;
- failure on unreadable images instead of silent substitution.

These changes improve the implementation but do not retroactively alter the historical 83.20% result.

## Limitations

- Validation was used during development and is not an untouched final test set.
- The crop split does not explicitly control spatial or source-scene correlation.
- The original checkpoint and per-image historical predictions are unavailable.
- Training accuracy under MixUp/CutMix is not directly comparable with validation accuracy.
- This repository covers satellite **classification**, not object detection.

See [`VALIDATION.md`](VALIDATION.md) for the exact checks executed during cleanup.
