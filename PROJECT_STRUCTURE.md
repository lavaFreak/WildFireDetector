# WildFireDetector Project Structure

## Root
- `README.md` — project overview + how to reproduce results
- `requirements.txt` — Python dependencies
- `.gitignore` — ignore data, models, caches
- `PROJECT_STRUCTURE.md` — quick orientation for repository layout

## Data
- `data/README.md` — expected data layout for the public-facing repo copy
- `data/raw/` — unmodified dataset after extraction (not committed in the public repo)
- `data/splits/` — canonical train/val/test structure created by scripts
  - `train/fire/`, `train/no_fire/`
  - `val/fire/`, `val/no_fire/`
  - `test/fire/`, `test/no_fire/`
  - `SPLIT_INFO.json` — split ratios, seed, counts, and file lists (reproducibility)

## Code
- `scripts/prepare_dataset.py` — builds `data/splits/` from `data/raw/`
- `scripts/summarize_results.py` — aggregates run outputs into Markdown summary tables
- `src/data_utils.py` — loads splits + builds flattened feature matrices
- `src/train_logreg.py` — trains logistic regression + saves metrics/plots

## Optional CNN
- `cnn/train_cnn.py` — CNN comparison experiment (secondary)

## Outputs
- `figures/` — plots for report (ROC curves, confusion matrices, weight heatmaps)
- `results/` — per-run `results.json` files plus aggregate summary tables
- `report/` — narrative write-up and project summary

## Tests
- `tests/test_data_utils.py` — synthetic tests for preprocessing and split loading
