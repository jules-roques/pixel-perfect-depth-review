import numpy as np
import torch


def image_array2tensor(image: np.ndarray) -> torch.Tensor:
    """
    Converts an RGB image array (H, W, 3) to a PyTorch tensor (3, H, W).
    """
    assert image.ndim == 3 and image.shape[-1] == 3, "Image must be (H, W, 3)"
    image = np.asarray(image).astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image).astype(np.float32)
    return torch.from_numpy(image)


def image_tensor2array(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor (3, H, W) to an RGB image array (H, W, 3).
    """
    assert tensor.ndim == 3 and tensor.shape[0] == 3, "Tensor must be (3, H, W)"
    image = tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.ascontiguousarray(image).astype(np.float32)
    return image


def linear_to_rgb(
    image: np.ndarray, exposure: float = 1.0, gamma: float = 2.2
) -> np.ndarray:
    """
    Converts a linear high dynamic range photometric image to a standard
    displayable RGB image in the [0, 1] range using gamma correction.
    """
    exposed_image = image * exposure
    tone_mapped_image = apply_reinhard_tone_mapping(exposed_image)
    gamma_corrected_image = apply_gamma_correction(tone_mapped_image, gamma)
    final_rgb_image = np.clip(gamma_corrected_image, 0.0, 1.0)
    return final_rgb_image


def apply_reinhard_tone_mapping(image: np.ndarray) -> np.ndarray:
    return image / (1.0 + image)


def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(image, 1.0 / gamma)
