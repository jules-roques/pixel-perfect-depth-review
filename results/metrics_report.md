# Model Evaluation Metrics

## Depth Score Metrics (δ = 1.25)

|    Model     |    Precision    |      Recall     |     F-score     |
| :----------: | :-------------: | :-------------: | :-------------: |
|     PPD      | 0.6564 ± 0.3609 | 0.6561 ± 0.3608 | 0.6563 ± 0.3609 |
|     DAv2     | 0.6553 ± 0.3377 | 0.6443 ± 0.3328 | 0.6495 ± 0.3349 |
| DAv2-Cleaned | 0.6493 ± 0.3361 | 0.6084 ± 0.3201 | 0.6277 ± 0.3272 |

## Per-Image Inference Time

|    Model     | Inference Time (ms) |
| :----------: | :-----------------: |
|     PPD      |   2317.50 ± 11.15   |
|     DAv2     |    290.24 ± 2.94    |
| DAv2-Cleaned |    369.03 ± 38.60   |

## Edge-Aware Chamfer Distance

|    Model     | Chamfer Distance |
| :----------: | :--------------: |
|     PPD      | 0.4695 ± 0.4397  |
|     DAv2     | 0.6192 ± 0.8278  |
| DAv2-Cleaned | 0.3941 ± 0.3324  |
