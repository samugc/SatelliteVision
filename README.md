# xView Computer Vision

13-class satellite crop classification with a pretrained Swin-Base transformer.
Originally developed during MSc Computer Vision coursework (2025–26), in a group
context. Samuel García confirms implementing the training and evaluation work.
The pretrained architecture and dataset are third-party resources.

## Approach
Full fine-tuning of `microsoft/swin-base-patch4-window7-224`, class-weighted
cross-entropy, MixUp/CutMix, mixed precision, gradient accumulation and four-view
TTA. This is image classification, not bounding-box object detection.

## Evidence
The original notebook records **83.20% validation accuracy with TTA** on 1,875
images. It is a historical output, not a reproduced result of the cleaned code.
`evidence/` contains source hashes, extracted outputs and a reconstructed split
(16,871 train / 1,875 validation). The annotation order was checked by image ID.
The original run did not preserve an annotation hash, so reconstruction cannot
prove that its input snapshot was identical. No Swin checkpoint or per-image
Swin predictions was located in the approved source folders.

The 84.27% attributed to Swin in slides is excluded: its model/output provenance
is ambiguous. No cross-model performance table is claimed. Detection slides
exist, but supporting implementation was not found in the scoped sources.

## Run
Python 3.11 recommended. From the repository root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
jupyter lab notebooks/swin_classification.ipynb
```
Obtain the original course xView recognition data under its applicable terms.
Set `XVIEW_DATA` to the directory containing `xview_ann_train.json` and
`xview_train/`. Images, weights and full annotations are not redistributed.
Do not run all training cells just to inspect the project; training requires
substantial compute. Initial model loading downloads pretrained weights.
The requirements specify compatibility ranges, not the unknown original lock.

Evaluate new exported predictions without training:
```bash
python -m classification.core evidence/validation_split.csv runs/swin/predictions.json
```
The evaluator requires complete image IDs and matching filenames, returning
accuracy, fixed-13-class macro-F1, per-class scores, confusion and error records.
Those metrics cannot be reconstructed for the historical Swin run from an
aggregate accuracy alone. Tests use synthetic records, not a claimed benchmark.

## 2026 portfolio changes
- Removed duplicate experiments and ambiguous index-based evaluation.
- Joined annotations and predictions by image ID and persisted split identities.
- Fixed the final gradient accumulation group and scheduler update count;
  scheduler advancement is skipped on AMP overflow.
- Fail on unreadable images instead of silently substituting black images.
- Export new predictions, metrics and checkpoint/annotation hashes.

These are cleanup/extension changes; **83.20% predates them**. Validation is used
for checkpoint selection and TTA inspection. It is not an untouched test set;
spatial/source-scene correlation is not controlled by this random crop split.
Training accuracy with MixUp/CutMix is not directly comparable to validation.

## Interview review
Explain class weighting plus label smoothing, Swin's local windows, augmentation
invariance for satellite images, TTA cost, spatial leakage and why a validation
score does not establish generalization. Review the AI-assisted 2026 edits,
especially accumulation normalization and identity checks, before claiming them
as personally mastered. No claim of reliable object detection or deployment.

Executed checks and limits: [VALIDATION.md](VALIDATION.md).
