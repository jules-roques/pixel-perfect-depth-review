from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EvalMetrics:
    inference_time: float
    chamfer_distance: float | None


@dataclass
class SafeDepth:
    depth: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class Intrinsics:
    """
    Camera intrinsics in pixel units.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, K: np.ndarray) -> "Intrinsics":
        """
        Create Intrinsics from a (3, 3) matrix.
        """
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )

    def to_matrix(self) -> np.ndarray:
        """
        Convert to a (3, 3) matrix.
        """
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K


@dataclass
class ImageContext:
    name: str
    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    gt_depth: SafeDepth
    intrinsics: Intrinsics
    device: torch.device
