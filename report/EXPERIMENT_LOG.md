# WildFireDetector Experiment Log

This file tracks the meaningful training and evaluation runs we have tried so far and what they taught us.

## Current Best

The new leader is the corrected outside-data fine-tuned checkpoint:

- `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune`
  - Best single checkpoint so far
  - Test accuracy at threshold `0.5`: `0.965`
  - Test AUC: `0.991`
  - Test precision / recall: `0.944 / 0.982`

This run also beats the older ensemble paths we had been using as the top result.

## Important Pipeline Fix

During evaluation of the earlier RGB checkpoints, we found that the original RGB training path reshaped flattened HWC image arrays directly into CHW tensors. That meant:

- the original RGB benchmark numbers were still internally valid, because train/val/test all shared the same layout
- inference on arbitrary images needed compatibility logic to preserve that legacy layout
- future RGB training runs should use an explicit channel-first layout instead

The repository now does both:

- old RGB checkpoints are still supported in `src/inference.py`
- new RGB checkpoints save `rgb_tensor_layout="chw"` metadata and use the corrected layout during training

## Completed Training Experiments

### Legacy grayscale baselines

| Run | Resolution | Accuracy | AUC | Notes |
|-----|------------|----------|-----|-------|
| `cnn_16x16` | 16x16 | 0.780 | 0.887 | Small grayscale CNN baseline |
| `cnn_64x64` | 64x64 | 0.879 | 0.942 | Strongest pre-augmentation grayscale baseline |
| `cnn_256x256` | 256x256 | 0.869 | 0.931 | Larger grayscale input did not outperform stronger RGB runs |

### Logistic regression baselines

| Run | Resolution | Accuracy | AUC | Notes |
|-----|------------|----------|-----|-------|
| `logreg_8x8` | 8x8 | 0.719 | 0.798 | Simple linear baseline |
| `logreg_16x16` | 16x16 | 0.719 | 0.800 | Slightly larger linear baseline |

### Improved CNN experiments

| Run | Resolution | Accuracy | AUC | Notes |
|-----|------------|----------|-----|-------|
| `cnn_64x64_rgb_aug_v2` | 64x64 | 0.932 | 0.979 | First strong RGB + augmentation run, using the legacy RGB tensor layout |
| `cnn_64x64_rgb_aug_v2_multifire20k` | 64x64 | 0.845 | 0.958 | Direct merge of MultiFire20K train data improved recall but hurt precision badly |
| `cnn_64x64_rgb_aug_v2_multifire20k_finetune` | 64x64 | 0.927 | 0.982 | MultiFire20K pretraining followed by fine-tuning improved ranking quality |
| `cnn_128x128_rgb_aug_v2` | 128x128 | 0.935 | 0.975 | Larger RGB input beat the original legacy `64x64 RGB` run but not the corrected `64x64` retrain |
| `cnn_64x64_rgb_aug_v2_chw` | 64x64 | 0.944 | 0.989 | Corrected channel-first RGB retrain; best single checkpoint before outside-data retraining was rebuilt |
| `cnn_64x64_rgb_aug_v2_chw_multifire20k` | 64x64 | 0.900 | 0.986 | Corrected MultiFire20K merged-data stage; strong ranking but too many false positives before fine-tuning |
| `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune` | 64x64 | 0.965 | 0.991 | Corrected MultiFire20K pretraining followed by in-domain fine-tuning; current best overall checkpoint |

## Combination Evaluations

These evaluations were produced with `scripts/evaluate_combo.py` on the preserved project validation/test split.

| Path | Default Test Accuracy | Tuned Test Accuracy | Test AUC | Notes |
|------|------------------------|---------------------|----------|-------|
| `cnn_64x64_rgb_aug_v2` | 0.932 | 0.929 | 0.979 | Legacy-layout RGB checkpoint, evaluated with compatibility mode |
| `cnn_64x64_rgb_aug_v2` + TTA | 0.936 | 0.932 | 0.979 | Small default-threshold bump from TTA |
| `cnn_64x64_rgb_aug_v2_multifire20k_finetune` | 0.927 | 0.932 | 0.982 | Best single-checkpoint AUC before the corrected retrain |
| `cnn_64x64_rgb_aug_v2 + cnn_64x64_rgb_aug_v2_multifire20k_finetune` | 0.935 | 0.945 | 0.982 | Old best ensemble before the corrected retrain |
| `cnn_64x64_rgb_aug_v2_chw` | 0.944 | 0.945 | 0.989 | Corrected in-domain baseline |
| `cnn_64x64_rgb_aug_v2_chw` + TTA | 0.945 | 0.946 | 0.989 | Small bump from TTA |
| `cnn_64x64_rgb_aug_v2_chw + cnn_64x64_rgb_aug_v2_multifire20k_finetune` | 0.952 | 0.952 | 0.989 | Previous best ensemble before the corrected fine-tune retrain |
| `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune` | 0.965 | 0.963 | 0.991 | New best single model; tuned threshold was not better than the default |
| `cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune` + TTA | 0.963 | 0.963 | 0.991 | TTA did not improve the new winner |
| `cnn_64x64_rgb_aug_v2_chw + cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune` | 0.959 | 0.956 | 0.991 | Ensemble with the corrected baseline underperformed the new single checkpoint |

## Lessons So Far

- RGB information matters a lot for this problem.
- Correcting the RGB channel layout improved the best single-checkpoint result substantially.
- Light augmentation plus a stronger architecture helps more than simply increasing grayscale resolution.
- Raw outside-data merging can hurt thresholded accuracy because of domain shift.
- Outside-data pretraining followed by fine-tuning on the original dataset is much more promising than direct permanent merging.
- Rebuilding the outside-data path on top of the corrected CHW baseline worked much better than keeping the legacy RGB checkpoint in that loop.
- The new corrected fine-tuned checkpoint is strong enough that ensembling is no longer necessary for the top score.

## Most Useful Next Experiments

1. Try a corrected `128x128 RGB` outside-data pretraining path only if we want to spend more compute on squeezing out incremental gains.
2. Add another outside dataset only after keeping the original benchmark split untouched and documenting source provenance.
3. If deployment behavior matters, calibrate thresholds on the validation split depending on whether we want higher recall or fewer false positives.
