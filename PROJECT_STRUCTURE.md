# WildFireDetector Project Structure

## Root
- `README.md` — project overview + how to reproduce results
- `requirements.txt` — Python dependencies
- `.gitignore` — ignore data, models, caches
- `PROJECT_STRUCTURE.md` — quick orientation for repository layout

## Data
- `data/README.md` — expected data layout
- `data/raw/` — unmodified dataset after extraction (not committed here)
- `data/splits/` — canonical train/val/test structure created by scripts
  - `train/fire/`, `train/no_fire/`
  - `val/fire/`, `val/no_fire/`
  - `test/fire/`, `test/no_fire/`
  - `SPLIT_INFO.json` — split ratios, seed, counts, and file lists (reproducibility)

## Code
- `app/classifier_app.py` — local desktop UI for choosing one image or a folder and running classification
- `scripts/prepare_dataset.py` — builds `data/splits/` from `data/raw/`
- `scripts/classify_images.py` — CLI for single-image or batch-image prediction
- `scripts/evaluate_combo.py` — evaluates single checkpoints or ensembles with optional TTA and threshold tuning
- `scripts/summarize_results.py` — aggregates run outputs into Markdown summary tables
- `src/data_utils.py` — loads splits + builds flattened feature matrices
- `src/inference.py` — checkpoint loading, preprocessing, and batch prediction helpers
- `src/eval_utils.py` — reusable probability-metric and threshold-selection helpers
- `src/train_logreg.py` — trains logistic regression + saves metrics/plots

## Optional CNN
- `cnn/train_cnn.py` — CNN comparison experiment (secondary)

## Outputs
- `figures/` — plots for report (ROC curves, confusion matrices, weight heatmaps)
- `results/` — per-run `results.json` files plus aggregate summary tables
  - `evaluations/` — saved single-model / ensemble evaluation JSON reports
- `report/` — narrative write-up and project summary
  - `EXPERIMENT_LOG.md` — record of completed training runs and observed outcomes
  - `MODEL_CARD.md` — intended use, metrics, and limitations for the strongest checkpoint
  - `PROJECT_BRIEF.md` — concise high-level project overview
  - `TRAINING_PLAN.md` — roadmap for stronger experiments and outside-dataset integration

## Tests
- `tests/test_data_utils.py` — synthetic tests for preprocessing and split loading
- `tests/test_eval_utils.py` — synthetic tests for thresholding and probability averaging helpers
- `tests/test_inference_utils.py` — synthetic tests for image discovery and inference preprocessing
