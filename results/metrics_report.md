# Model Evaluation Metrics

## Depth Score Metrics (δ = 1.25)

|    Model     |    Precision    |      Recall     |     F-score     |
| :----------: | :-------------: | :-------------: | :-------------: |
|     PPD      | 0.5098 ± 0.3960 | 0.5096 ± 0.3959 | 0.5097 ± 0.3960 |
|     DAv2     | 0.5062 ± 0.3783 | 0.4987 ± 0.3725 | 0.5023 ± 0.3751 |
| DAv2-Cleaned | 0.4944 ± 0.3607 | 0.4644 ± 0.3415 | 0.4786 ± 0.3502 |

## Per-Image Inference Time

|    Model     | Inference Time (ms) |
| :----------: | :-----------------: |
|     PPD      |    2363.89 ± 9.57   |
|     DAv2     |    309.69 ± 2.08    |
| DAv2-Cleaned |    420.80 ± 44.51   |

## Edge-Aware Chamfer Distance

|    Model     | Chamfer Distance |
| :----------: | :--------------: |
|     PPD      | 4.1634 ± 8.2314  |
|     DAv2     | 4.2952 ± 9.0333  |
| DAv2-Cleaned | 5.0711 ± 10.8197 |
