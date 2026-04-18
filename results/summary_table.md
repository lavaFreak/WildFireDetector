| Model | Run | Variant | Resolution | Accuracy | AUC | FP | FN |
|-------|-----|---------|------------|----------|-----|----|----|
| CNN | cnn_16x16 | cnn_16x16 | 16x16 | 0.780 | 0.887 | 108 | 48 |
| CNN | cnn_64x64 | cnn_64x64 | 64x64 | 0.879 | 0.942 | 41 | 45 |
| CNN | cnn_64x64_rgb_aug_v2 | rgb-wildfire_cnn_v2-aug | 64x64 | 0.932 | 0.979 | 10 | 38 |
| CNN | cnn_64x64_rgb_aug_v2_chw | rgb-wildfire_cnn_v2-aug | 64x64 | 0.944 | 0.989 | 26 | 14 |
| CNN | cnn_64x64_rgb_aug_v2_chw_multifire20k | rgb-wildfire_cnn_v2-aug | 64x64 | 0.900 | 0.986 | 68 | 3 |
| CNN | cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune | rgb-wildfire_cnn_v2-aug | 64x64 | 0.965 | 0.991 | 19 | 6 |
| CNN | cnn_64x64_rgb_aug_v2_multifire20k | rgb-wildfire_cnn_v2-aug | 64x64 | 0.845 | 0.958 | 98 | 12 |
| CNN | cnn_64x64_rgb_aug_v2_multifire20k_finetune | rgb-wildfire_cnn_v2-aug | 64x64 | 0.927 | 0.982 | 13 | 39 |
| CNN | cnn_128x128_rgb_aug_v2 | rgb-wildfire_cnn_v2-aug | 128x128 | 0.935 | 0.975 | 13 | 33 |
| CNN | cnn_256x256 | cnn_256x256 | 256x256 | 0.869 | 0.931 | 42 | 51 |
| LogReg | logreg_8x8 | logreg_8x8 | 8x8 | 0.719 | 0.798 | 88 | 111 |
| LogReg | logreg_16x16 | logreg_16x16 | 16x16 | 0.719 | 0.800 | 87 | 112 |
