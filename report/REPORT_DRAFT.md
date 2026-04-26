# WildFireDetector Report Notes

## Goal

Build a binary image classifier that distinguishes wildfire images from no-fire images, compare a simple linear baseline against a small CNN, and document the tradeoffs between model complexity and performance.

## Background

This project grew out of earlier wildfire-detection work I contributed to during an internship with Other Orb LLC. For a later class project, I adapted that general problem into a more self-contained experiment by adding logistic-regression baselines and organizing the workflow around reproducible dataset preparation, training, and result summaries.

This repository adapts that work without exposing proprietary code or internal materials.

## Workflow

1. Collect raw wildfire and no-fire images.
2. Build deterministic train, validation, and test splits with `scripts/prepare_dataset.py`.
3. Train logistic-regression baselines on downsampled grayscale inputs.
4. Train a small CNN on image tensors for a stronger comparison model.
5. Save metrics, ROC curves, confusion matrices, and summary tables for analysis.

## Current Results

| Model | Resolution | Accuracy | AUC | Precision | Recall | F1 |
|-------|------------|----------|-----|-----------|--------|----|
| Best CNN (`cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune`) | 64x64 RGB | 0.965 | 0.991 | 0.944 | 0.982 | 0.963 |
| Corrected in-domain CNN (`cnn_64x64_rgb_aug_v2_chw`) | 64x64 RGB | 0.944 | 0.989 | 0.924 | 0.957 | 0.940 |
| Legacy grayscale CNN | 64x64 gray | 0.879 | 0.942 | 0.874 | 0.863 | 0.869 |
| LogReg baseline | 16x16 gray | 0.719 | 0.800 | 0.714 | 0.660 | 0.686 |

The project now shows a fuller progression than the original class version:

- simple linear baselines underperform because flattening throws away spatial structure
- moving to RGB plus augmentation improves performance substantially
- rebuilding the training path after fixing RGB tensor layout improves the strongest single-checkpoint score again
- outside-data pretraining followed by in-domain fine-tuning yields the best final model

## Takeaways

- Logistic regression provides a useful baseline, but it loses too much spatial information when the images are flattened.
- The CNN captures stronger visual features and improves both discrimination and false-positive control.
- Correct preprocessing mattered materially: fixing the RGB tensor layout improved the best single-checkpoint result.
- Outside data helped most when it was used for pretraining and then followed by in-domain fine-tuning rather than merged permanently into the benchmark setting.
- The pipeline is structured to make experiments reproducible: fixed split generation, saved run artifacts, and aggregate summaries under `results/`.
- Including the logistic-regression baseline makes the CNN results more meaningful, because the project shows why the additional model complexity is worth using.

## Limitations

- The original dataset is not included in this repository.
- The project currently focuses on offline experimentation, not deployment or real-time inference.
- Model and dataset provenance should be documented more explicitly before treating the project as a research-grade benchmark.

## Next Steps

- Document dataset provenance and licensing more precisely.
- If more experimentation is needed, test corrected larger-input outside-data training rather than random hyperparameter changes.
- Continue improving dataset provenance notes and evaluation clarity rather than chasing small metric gains for their own sake.
