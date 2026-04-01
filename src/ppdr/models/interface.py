from abc import ABC, abstractmethod

import torch


class DepthModel(ABC):
    @abstractmethod
    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        pass
