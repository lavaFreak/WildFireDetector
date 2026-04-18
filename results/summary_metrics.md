| Model | Run | Variant | Resolution | Precision | Recall | F1 | TP | TN | FP | FN |
|-------|-----|---------|------------|-----------|--------|----|----|----|----|----|
| CNN | cnn_16x16 | cnn_16x16 | 16x16 | 0.722 | 0.854 | 0.783 | 281 | 271 | 108 | 48 |
| CNN | cnn_64x64 | cnn_64x64 | 64x64 | 0.874 | 0.863 | 0.869 | 284 | 338 | 41 | 45 |
| CNN | cnn_64x64_rgb_aug_v2 | rgb-wildfire_cnn_v2-aug | 64x64 | 0.967 | 0.884 | 0.924 | 291 | 369 | 10 | 38 |
| CNN | cnn_64x64_rgb_aug_v2_chw | rgb-wildfire_cnn_v2-aug | 64x64 | 0.924 | 0.957 | 0.940 | 315 | 353 | 26 | 14 |
| CNN | cnn_64x64_rgb_aug_v2_chw_multifire20k | rgb-wildfire_cnn_v2-aug | 64x64 | 0.827 | 0.991 | 0.902 | 326 | 311 | 68 | 3 |
| CNN | cnn_64x64_rgb_aug_v2_chw_multifire20k_finetune | rgb-wildfire_cnn_v2-aug | 64x64 | 0.944 | 0.982 | 0.963 | 323 | 360 | 19 | 6 |
| CNN | cnn_64x64_rgb_aug_v2_multifire20k | rgb-wildfire_cnn_v2-aug | 64x64 | 0.764 | 0.964 | 0.852 | 317 | 281 | 98 | 12 |
| CNN | cnn_64x64_rgb_aug_v2_multifire20k_finetune | rgb-wildfire_cnn_v2-aug | 64x64 | 0.957 | 0.881 | 0.918 | 290 | 366 | 13 | 39 |
| CNN | cnn_128x128_rgb_aug_v2 | rgb-wildfire_cnn_v2-aug | 128x128 | 0.958 | 0.900 | 0.928 | 296 | 366 | 13 | 33 |
| CNN | cnn_256x256 | cnn_256x256 | 256x256 | 0.869 | 0.845 | 0.857 | 278 | 337 | 42 | 51 |
| LogReg | logreg_8x8 | logreg_8x8 | 8x8 | 0.712 | 0.663 | 0.687 | 218 | 291 | 88 | 111 |
| LogReg | logreg_16x16 | logreg_16x16 | 16x16 | 0.714 | 0.660 | 0.686 | 217 | 292 | 87 | 112 |
