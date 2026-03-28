import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
model = make_pipeline(poly_features, ransac)


def recover_metric_depth_ransac(pred, gt, mask, log=True):
    """
    Recover metric depth from predicted depth using RANSAC.

    Args:
        pred: Predicted depth map.
        gt: Ground truth depth map.
        mask: Mask indicating valid depth values.
        log: Whether to use log depth.

    Returns:
        Recovered metric depth map.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mask_gt = gt[mask].astype(np.float32)
    ori_mask_gt = mask_gt
    mask_pred = pred[mask].astype(np.float32)

    ## depth -> log depth
    mask_gt = np.log(mask_gt + 1.0)

    try:
        model.fit(mask_pred[:, None], mask_gt[:, None])
        a, b = (
            model.named_steps["ransacregressor"].estimator_.coef_,
            model.named_steps["ransacregressor"].estimator_.intercept_,
        )
        a = a.item()
        b = b.item()
    except:
        a, b = 1, 0

    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(mask_pred)
        gt_mean = np.mean(mask_gt)
        pred_metric = pred * (gt_mean / pred_mean)

    ## log depth -> depth
    pred_metric = np.exp(pred_metric) - 1.0
    pred_metric = np.clip(pred_metric, 1e-3, np.max(ori_mask_gt))
    return pred_metric
