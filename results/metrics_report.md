# Model Evaluation Metrics

## Depth Score Metrics (δ = 1.05)

|    Model     |    Precision    |      Recall     |     F-score     |
| :----------: | :-------------: | :-------------: | :-------------: |
|     PPD      | 0.6368 ± 0.2654 | 0.6404 ± 0.2664 | 0.6384 ± 0.2656 |
|     DAv2     | 0.5932 ± 0.2502 | 0.5871 ± 0.2471 | 0.5898 ± 0.2478 |
| DAv2-Cleaned | 0.6142 ± 0.2514 | 0.5769 ± 0.2391 | 0.5943 ± 0.2436 |

## Per-Image Inference Time

|    Model     | Inference Time (ms) |
| :----------: | :-----------------: |
|     PPD      |   3783.03 ± 504.33  |
|     DAv2     |    299.14 ± 32.09   |
| DAv2-Cleaned |    372.11 ± 38.81   |

## Edge-Aware Chamfer Distance

|    Model     | Chamfer Distance |
| :----------: | :--------------: |
|     PPD      | 0.2740 ± 0.2151  |
|     DAv2     | 0.5070 ± 0.5162  |
| DAv2-Cleaned | 0.2285 ± 0.2453  |
