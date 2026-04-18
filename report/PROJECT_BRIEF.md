# WildFireDetector Project Brief

## One-Sentence Summary

WildFireDetector is a public-facing wildfire image-classification project that compares linear baselines against CNNs, adds reproducible dataset handling and evaluation tooling, and improves the best model to `96.5%` test accuracy through corrected RGB preprocessing plus outside-data pretraining and in-domain fine-tuning.

## Problem

Given an input image, classify it as either:

- `fire`
- `no_fire`

The project focuses on binary image classification rather than detection or segmentation, which keeps the modeling goal narrow and makes baseline-vs-CNN comparisons easier to interpret.

## Why This Project Is Stronger Than A Typical Class Project

- It starts with simple baselines instead of jumping straight to a CNN.
- It preserves a fixed validation/test benchmark while experimenting.
- It documents unsuccessful ideas, not just the winning run.
- It includes an inference path for arbitrary user-selected images.
- It surfaced and fixed a real RGB preprocessing bug during evaluation.

## My Role

- Adapted an earlier wildfire-classification problem into a standalone public repository.
- Added logistic-regression baselines for comparison.
- Reworked the training and evaluation pipeline to be more reproducible.
- Added support for arbitrary-image inference through a CLI and local app.
- Investigated image-size, augmentation, and outside-dataset tradeoffs.
- Rebuilt the strongest training path after discovering an RGB tensor-layout issue.

## Best Result

Best checkpoint:

- `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune`
- Accuracy: `0.965`
- AUC: `0.991`
- Precision: `0.944`
- Recall: `0.982`
- F1: `0.963`

## Key Engineering Decisions

1. Keep the original project validation/test split unchanged while adding external data only to training stages.
2. Compare simple linear baselines against CNNs to justify added model complexity.
3. Treat outside datasets as a robustness tool, not as permission to contaminate the benchmark.
4. Preserve compatibility with legacy checkpoints while fixing the RGB layout for future training.

## Lessons Learned

- RGB information mattered much more than grayscale-only scaling.
- Larger input size alone was not the main source of gains.
- Directly merging outside data hurt thresholded accuracy because of domain shift.
- Pretraining on outside data and then fine-tuning back on the in-domain split worked much better.
- A clean preprocessing pipeline can matter as much as architecture changes.

## Good Interview Talking Points

- Why logistic regression was still worth including.
- Why `256x256` grayscale did not help as much as corrected `64x64` RGB.
- How the RGB tensor-layout bug was discovered and fixed.
- Why direct dataset mixing hurt precision and why fine-tuning improved it.
- How the project was adapted into a public portfolio piece without exposing proprietary materials.

## Related Docs

- [README.md](../README.md)
- [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)
- [MODEL_CARD.md](./MODEL_CARD.md)
