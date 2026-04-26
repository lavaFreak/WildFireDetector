# WildFireDetector

WildFireDetector is a binary image-classification project that compares simple logistic-regression baselines against a small convolutional neural network for wildfire vs. no-fire detection.

The project is structured around a reproducible workflow:
- prepare a canonical train/validation/test split from raw images
- train baseline logistic-regression models on downsampled grayscale features
- train a small CNN for comparison
- save metrics, ROC curves, confusion matrices, and summary tables for analysis

## Project Background

This repository is a cleaned-up adaptation of earlier wildfire-detection work I contributed to during an internship with Other Orb LLC, followed by a class-project extension where I added a logistic-regression baseline and a more explicit comparison pipeline.

The goal of this version is to keep the project reproducible and the modeling comparison explicit:
- preserve the core wildfire-classification problem
- show the difference between a simple baseline and a stronger CNN approach

This repository is not a mirror of internship code. It does not include proprietary code, private datasets, or internal company materials.

## Highlights

- Reproducible dataset-splitting script with explicit labeling rules and split metadata
- Baseline comparison across logistic regression and CNN models
- Saved figures and JSON outputs for each experiment run
- Lightweight data-loading utilities and tests that do not require the original dataset

## Start Here

- [Project Brief](report/PROJECT_BRIEF.md): fastest high-level overview of the problem, approach, and best result
- [Model Card](report/MODEL_CARD.md): intended use, benchmark numbers, and limitations for the best checkpoint
- [Experiment Log](report/EXPERIMENT_LOG.md): full record of what was tried and what worked

## Results Snapshot

The strongest checkpoint in the current project copy is the corrected `64x64 RGB + augmentation + wildfire_cnn_v2` model after MultiFire20K pretraining and in-domain fine-tuning:

| Path | Resolution | Accuracy | AUC | Precision | Recall | F1 |
|------|------------|----------|-----|-----------|--------|----|
| `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune` | 64x64 | 0.965 | 0.991 | 0.944 | 0.982 | 0.963 |

This run currently beats every other single checkpoint and every ensemble we tried on default-threshold test accuracy.

Additional training runs are summarized in [results/summary_table.md](results/summary_table.md) and [results/summary_metrics.md](results/summary_metrics.md). Reproducible combination and threshold-tuning evaluations are saved under [results/evaluations](results/evaluations).

Recent experiment notes:

- `256x256` grayscale did not outperform the stronger RGB runs, so larger grayscale input alone is not the main improvement path for this project.
- `128x128 RGB` improved over the original legacy `64x64 RGB` run, but the corrected `64x64 RGB` retrain still produced the best single-checkpoint score.
- Rebuilding the MultiFire20K pretraining path on top of the corrected CHW RGB baseline produced the best overall result so far.

## Why Include Logistic Regression?

The logistic-regression models are there on purpose. I added them in the class-project adaptation to establish a simple baseline before comparing against the CNN.

That comparison makes the project stronger because it answers a more useful question than "can a CNN classify these images?":
- how much performance do we get from a simple linear baseline?
- when does a higher-capacity model justify its extra complexity?
- what do we lose when we flatten the image and remove most spatial structure?

In the current results, the CNN outperforms the logistic-regression baselines clearly, which helps justify the move to the more expressive model.

## Example Artifacts

### CNN ROC Curve

![CNN ROC](figures/cnn_64x64/test_roc.png)

### CNN Confusion Matrix

![CNN confusion matrix](figures/cnn_64x64/test_confusion_matrix.png)

## Interactive Classification

The repository now includes two ways to classify arbitrary images with the saved CNN checkpoints:

1. Batch CLI for one image, many images, or entire folders:

```bash
PYTHONPATH=. python scripts/classify_images.py path/to/image_or_folder --model cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune
```

2. Local desktop app with file/folder selection:

```bash
PYTHONPATH=. python app/classifier_app.py
```

The default classifier is the corrected `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune` checkpoint. The `ensemble` option is still available for comparison, but the strongest score in the repo now comes from the single corrected fine-tuned checkpoint.

Quick smoke-test command:

```bash
PYTHONPATH=. python scripts/classify_images.py figures/cnn_64x64/test_roc.png
```

Compatibility note:

- earlier RGB checkpoints used a legacy tensor layout during training
- the inference code now preserves compatibility with those saved checkpoints
- new RGB checkpoints are trained with an explicit channel-first layout and store that metadata in the checkpoint

## Repository Layout

- `scripts/prepare_dataset.py`: builds canonical `train/`, `val/`, and `test/` splits from raw images
- `src/data_utils.py`: loads image splits and creates feature matrices
- `src/train_logreg.py`: trains logistic-regression baselines and saves metrics plus figures
- `cnn/train_cnn.py`: trains the CNN comparison model
- `scripts/evaluate_combo.py`: evaluates one or more saved checkpoints with optional TTA and validation-tuned threshold selection
- `results/`: saved `results.json` files and aggregate summary tables
- `figures/`: ROC curves, confusion matrices, learning curves, and weight visualizations
- `tests/`: lightweight tests for data-loading and preprocessing behavior

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the CNN experiments, install PyTorch separately using the instructions for your platform from [pytorch.org](https://pytorch.org/).

## Reproducing The Pipeline

1. Prepare the dataset splits:

```bash
python scripts/prepare_dataset.py --input data/raw --output data/splits --val 0.15 --test 0.15 --seed 42 --mode copy --keep-unknown
```

2. Train the logistic-regression baseline:

```bash
python -m src.train_logreg --size 16
```

3. Train the CNN comparison model:

```bash
python cnn/train_cnn.py --size 64 --epochs 25
```

Recommended stronger run:

```bash
python cnn/train_cnn.py --size 64 --rgb --augment --arch wildfire_cnn_v2 --run-name rgb_aug_v2_chw --epochs 20
```

Current strongest training path:

```bash
python cnn/train_cnn.py --size 64 --rgb --augment --arch wildfire_cnn_v2 --run-name rgb_aug_v2_chw_multifire20k --extra-train-root /path/to/multifire20k/canonical --init-checkpoint figures/cnn_64x64_rgb_aug_v2_chw/best_model.pt --epochs 15
python cnn/train_cnn.py --size 64 --rgb --augment --arch wildfire_cnn_v2 --run-name rgb_aug_v2_chw_multifire20k_finetune --init-checkpoint figures/cnn_64x64_rgb_aug_v2_chw_multifire20k/best_model.pt --lr 0.0003 --epochs 10
```

4. Regenerate the summary tables:

```bash
python scripts/summarize_results.py
```

5. Evaluate a saved checkpoint or ensemble with optional TTA:

```bash
python scripts/evaluate_combo.py --name rgb_chw_multifire_chw_finetune_eval --checkpoint figures/cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune/best_model.pt --splits-root data/splits
```

## Testing

The included tests use temporary synthetic image data, so they can run without the original wildfire dataset:

```bash
PYTHONPATH=. python -m pytest tests
```

The new inference utilities are also covered with lightweight tests for file discovery and preprocessing behavior.

## Training Roadmap

The concrete next-step plan for augmentation, RGB experiments, and outside-dataset expansion is documented in [report/TRAINING_PLAN.md](report/TRAINING_PLAN.md).

The running record of completed experiments and what they taught us is tracked in [report/EXPERIMENT_LOG.md](report/EXPERIMENT_LOG.md).

For a concise summary, see [report/PROJECT_BRIEF.md](report/PROJECT_BRIEF.md). For checkpoint-specific limitations and intended use, see [report/MODEL_CARD.md](report/MODEL_CARD.md).

## Data Note

The original dataset is not included here. See [data/README.md](data/README.md) for the expected directory layout and how the project uses `data/raw/` and `data/splits/`.
