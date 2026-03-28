import cv2
import numpy as np
import torch

from ppdr.utils.types import Intrinsics
from ppdr.vendor.ppd.moge.model.v2 import MoGeModel as VendorMoGe


class MoGe:
    def __init__(self, model_path: str, device: torch.device) -> None:
        self.model = VendorMoGe.from_pretrained(model_path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def infer_intrinsics(self, bgr: np.ndarray) -> Intrinsics:
        """
        Process a BGR image to produce intrinsics.
        """

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(
            rgb / 255.0, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        _, _, intrinsics = self.model.infer(tensor)

        intrinsics[0, 0] *= w
        intrinsics[1, 1] *= h
        intrinsics[0, 2] *= w
        intrinsics[1, 2] *= h

        return Intrinsics.from_matrix(intrinsics)
