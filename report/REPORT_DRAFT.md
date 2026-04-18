# WildFireDetector Report Notes

## Goal

Build a binary image classifier that distinguishes wildfire images from no-fire images, compare a simple linear baseline against a small CNN, and document the tradeoffs between model complexity and performance.

## Background

This project grew out of earlier wildfire-detection work I contributed to during an internship with Other Orb LLC. For a later class project, I adapted that general problem into a more self-contained experiment by adding logistic-regression baselines and organizing the workflow around reproducible dataset preparation, training, and result summaries.

This public repository is a presentation-oriented adaptation of that work. It is meant to show the engineering workflow and modeling comparison clearly without exposing proprietary code or internal materials.

## Workflow

1. Collect raw wildfire and no-fire images.
2. Build deterministic train, validation, and test splits with `scripts/prepare_dataset.py`.
3. Train logistic-regression baselines on downsampled grayscale inputs.
4. Train a small CNN on image tensors for a stronger comparison model.
5. Save metrics, ROC curves, confusion matrices, and summary tables for analysis.

## Current Results

| Model | Resolution | Accuracy | AUC | Precision | Recall | F1 |
|-------|------------|----------|-----|-----------|--------|----|
| CNN | 16x16 | 0.780 | 0.887 | 0.722 | 0.854 | 0.783 |
| CNN | 64x64 | 0.879 | 0.942 | 0.874 | 0.863 | 0.869 |
| LogReg | 8x8 | 0.719 | 0.798 | 0.712 | 0.663 | 0.687 |
| LogReg | 16x16 | 0.719 | 0.800 | 0.714 | 0.660 | 0.686 |

The CNN at `64x64` clearly outperforms the logistic-regression baselines in both accuracy and AUC, which suggests that retaining more spatial structure matters for this task.

## Takeaways

- Logistic regression provides a useful baseline, but it loses too much spatial information when the images are flattened.
- The CNN captures stronger visual features and improves both discrimination and false-positive control.
- The pipeline is structured to make experiments reproducible: fixed split generation, saved run artifacts, and aggregate summaries under `results/`.
- Including the logistic-regression baseline makes the CNN results more meaningful, because the project shows why the additional model complexity is worth using.

## Limitations

- The original dataset is not included in this public repository copy.
- The project currently focuses on offline experimentation, not deployment or real-time inference.
- Model and dataset provenance should be documented more explicitly before treating the project as a research-grade benchmark.

## Next Steps

- Document dataset provenance and licensing more precisely.
- Add a lightweight inference script for single-image prediction.
- Expand automated tests beyond data loading into metrics and training-output validation.
