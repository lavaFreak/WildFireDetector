# WildFireDetector Model Card

## Model Name

`cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune`

## Model Type

Binary image classifier for:

- `fire`
- `no_fire`

Architecture:

- `wildfire_cnn_v2`
- RGB input
- `64x64` resolution

Training recipe:

1. Start from the corrected CHW RGB baseline trained on the original project split.
2. Continue training with additional `MultiFire20K` data merged into the training split only.
3. Fine-tune the resulting checkpoint back on the original project split.

## Primary Metrics

Held-out project test split:

- Accuracy: `0.965`
- AUC: `0.991`
- Precision: `0.944`
- Recall: `0.982`
- F1: `0.963`

Confusion matrix:

- TN: `360`
- FP: `19`
- FN: `6`
- TP: `323`

## Intended Use

This model is appropriate for:

- experimentation with wildfire-style image classification
- experimentation with wildfire image classification workflows
- evaluating model iteration, benchmarking, and domain-shift handling

This model is not presented as:

- a production emergency-response system
- a real-time wildfire detection service
- a research-grade benchmark suitable for safety-critical deployment

## Input Expectations

- standard image file input
- RGB image preprocessing
- resized to `64x64`
- binary wildfire-vs-no-fire framing

The repository supports arbitrary image classification through:

- [scripts/classify_images.py](../scripts/classify_images.py)
- [app/classifier_app.py](../app/classifier_app.py)

## Benchmark Discipline

- The original project validation/test split was kept fixed while outside data was introduced.
- External data was used only in training/pretraining stages, not to redefine the benchmark.
- Combination and threshold experiments are logged under [results/evaluations](../results/evaluations).

## Known Limitations

- The dataset used for the original project is not distributed in this repository.
- Results may not transfer cleanly to very different domains such as satellite-only or smoke-only imagery.
- The project focuses on classification, not localization or segmentation.
- The benchmark is useful for project evaluation, but it is not a deployment guarantee.

## Important Historical Note

Earlier RGB checkpoints in this repository used a legacy tensor-layout path during training. The repository now:

- keeps compatibility with those older checkpoints for inference
- trains new RGB checkpoints with explicit channel-first layout metadata

The model in this card is part of the corrected CHW training path.
