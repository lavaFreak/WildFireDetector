# Data Layout

This public-facing copy does not include the original wildfire image dataset.

The project expects the following directories when reproducing the full pipeline:

- `data/raw/`: the unmodified source dataset after extraction
- `data/splits/`: canonical train, validation, and test folders created by `scripts/prepare_dataset.py`

Expected split layout:

```text
data/splits/
  train/
    fire/
    no_fire/
  val/
    fire/
    no_fire/
  test/
    fire/
    no_fire/
```

The preparation script also writes `data/splits/SPLIT_INFO.json` with split counts, seed values, and source file metadata for reproducibility.
