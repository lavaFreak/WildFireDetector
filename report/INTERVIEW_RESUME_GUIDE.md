# WildFireDetector Resume And Interview Guide

## Resume Bullet Options

Choose the version that best fits the amount of space you have.

### Short

- Built a public wildfire image-classification project with reproducible training/evaluation pipelines, improving the best model to `96.5%` test accuracy and `0.991` AUC.

### Medium

- Adapted a wildfire image-classification project into a public repository with reproducible dataset splitting, baseline comparison, arbitrary-image inference, and experiment tracking; improved the best model to `96.5%` test accuracy and `0.991` AUC.

### Detailed

- Built a public wildfire image-classification project comparing logistic-regression baselines against CNNs, added reproducible dataset preparation and evaluation tooling, discovered and fixed an RGB preprocessing bug, and improved the final model to `96.5%` test accuracy and `0.991` AUC through outside-data pretraining plus in-domain fine-tuning.

## 30-Second Interview Version

"I took a wildfire image-classification problem and turned it into a reproducible public project. I started with logistic-regression baselines, moved to CNNs, built tooling for train/validation/test benchmarking and arbitrary-image inference, and then improved the final model to `96.5%` accuracy. One of the most useful parts was finding and fixing an RGB preprocessing bug and rebuilding the stronger training path on top of that correction."

## 60-Second Interview Version

"This project started as an adaptation of earlier wildfire-detection work, and I turned it into a public repo that I could actually defend in interviews. I kept the workflow reproducible with fixed train/validation/test splits, compared logistic regression against CNNs so the stronger model had a real baseline, and experimented with image size, augmentation, and outside-data training. A big turning point was discovering that the original RGB path had a tensor-layout mismatch between training and inference. I fixed that, retrained the corrected model, and then used MultiFire20K as outside-data pretraining before fine-tuning back on the original benchmark split. That got the final model to `96.5%` accuracy with `0.991` AUC."

## Good Interview Themes

- baseline comparison instead of only showing the final model
- reproducibility and benchmark discipline
- debugging a data/preprocessing bug instead of only tuning hyperparameters
- careful handling of external data and domain shift
- improving a project into something public-facing and demoable

## Questions You Should Be Ready For

### Why include logistic regression?

To establish a simple baseline and show why the CNN was worth the added complexity.

### Why not just keep increasing image size?

Because larger grayscale inputs alone did not meaningfully improve results. RGB information and the corrected preprocessing path mattered more than raw size.

### Why did outside data help only after fine-tuning?

Directly merging outside data increased recall but introduced too many false positives. Pretraining on outside data and then fine-tuning on the in-domain split preserved the ranking benefit while recovering better thresholded accuracy.

### What was the hardest bug?

The RGB tensor-layout mismatch. The original RGB benchmark looked strong internally, but inference on arbitrary images needed compatibility logic because the training path had reshaped image data incorrectly.

## Best Supporting Files

- [README.md](../README.md)
- [PROJECT_BRIEF.md](./PROJECT_BRIEF.md)
- [MODEL_CARD.md](./MODEL_CARD.md)
- [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)
