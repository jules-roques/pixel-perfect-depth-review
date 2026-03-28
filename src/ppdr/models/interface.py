from abc import ABC, abstractmethod

import numpy as np


class DepthModel(ABC):
    @abstractmethod
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        pass
