# WildFireDetector Training Plan

## Current Baseline

Best current model in this repository:

- `CNN 64x64 RGB + augmentation + wildfire_cnn_v2`
- Accuracy: `0.932`
- AUC: `0.979`
- Precision: `0.967`
- Recall: `0.884`
- F1: `0.924`

Important recent result:

- `CNN 256x256 grayscale` did **not** beat the `64x64` model.
- That suggests input resolution alone is not the main bottleneck for this project.

## Main Goal

Improve accuracy and robustness without contaminating evaluation.

That means:

1. Keep the current project test split unchanged as the main comparison benchmark.
2. Improve the training pipeline before adding complexity for its own sake.
3. Add outside datasets carefully so they help generalization instead of causing domain-shift regressions.

## Priority Order

### Phase 1: Better Single-Dataset Training

Use the current dataset only, but improve the training procedure.

Experiments:

1. `64x64 RGB + augmentation + wildfire_cnn_v2` as the new strong baseline
2. `128x128 RGB + augmentation + wildfire_cnn_v2`
3. `64x64 RGB + augmentation + wildfire_cnn_v2` with outside-dataset pretraining

Why:

- `64x64 RGB` is already strong, so it is the best base to improve from.
- RGB may carry useful fire-color information that grayscale loses.
- The upgraded CNN and augmentation should help more than simply increasing to `256x256`.

Success criterion:

- Beat the current `64x64 RGB + augmentation` baseline on test AUC and/or reduce false negatives meaningfully.

### Phase 2: Controlled Multi-Dataset Expansion

After Phase 1, introduce external datasets for more varied training examples.

Recommended order:

1. Add `FlameVision`
2. Add `Forest Fire Dataset`
3. Optionally add selected `D-Fire` negatives or smoke-adjacent examples

Rules:

- Do **not** mix outside data into the existing test split.
- Keep the current project validation/test split untouched.
- Treat outside datasets primarily as extra training data.
- If needed, reserve a small validation slice from each outside dataset for source-specific checks.

Why this order:

- `FlameVision` is the closest high-value classification dataset match.
- `Forest Fire Dataset` is also a clean fire-vs-no-fire classification source.
- `D-Fire` is useful, but it is more smoke/detection oriented, so it should be added carefully.

### Phase 3: Domain-Shift Evaluation

Once multi-dataset training works, evaluate three ways:

1. Original project test split
2. External-source held-out validation
3. Cross-source spot checks through the classifier app / CLI

This helps answer:

- Did the model become genuinely better?
- Or did it just become better at one dataset style and worse at another?

## Concrete Experiment Matrix

Recommended next runs:

1. `64x64 grayscale`, `wildfire_cnn_v2`, augmentation enabled
2. `64x64 RGB`, `wildfire_cnn_v2`, augmentation enabled
3. `128x128 RGB`, `wildfire_cnn_v2`, augmentation enabled
4. Best of the above + `FlameVision` as additional training data

## Notes On Dataset Mixing

When adding outside datasets:

- normalize labels to a strict binary scheme: `fire` vs `no_fire`
- document which source each image came from
- avoid duplicating near-identical images across sources
- keep a manifest of train/val/test origin
- if a dataset includes smoke-only images, decide explicitly whether they belong in `fire` or should be excluded for this binary project

## Practical Recommendation

Do **not** spend more time on larger grayscale resolutions right now.

The best next use of time is:

1. augmentation
2. RGB experiments
3. better checkpointing / cleaner model architecture
4. outside datasets only after that baseline improves
