from abc import ABC, abstractmethod

import torch


class DepthModel(ABC):
    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def predict(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert rgb.ndim == 4 and rgb.shape[1] == 3 and rgb.dtype == torch.float32, (
            "Input must be (B, 3, H, W) with dtype float32"
        )
        pred, mask = self._predict(rgb)
        assert pred.ndim == 3 and pred.dtype == torch.float32, (
            "Output must be (B, H, W) with dtype float32"
        )
        assert mask.ndim == 3 and mask.dtype == torch.bool, (
            "Output mask must be (B, H, W) with dtype bool"
        )
        return pred, mask

    @abstractmethod
    def _predict(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def align_pred_on_metric_depth(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pass
