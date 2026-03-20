"""
Hypersim HDF5 + image data loader for the benchmark.
"""

import os
import cv2
import h5py
import numpy as np
from typing import Iterator, Tuple, Optional


class HypersimLoader:
    """
    Iterate over downloaded Hypersim test-set folders.

    Each folder contains:
        - ``frame.color.jpg``          (RGB preview)
        - ``frame.depth_meters.hdf5``  (Euclidean range depth)

    Yields ``(image_bgr, gt_depth_euclidean, valid_mask, entry_name)``
    tuples.
    """

    def __init__(self, data_root: str, max_images: Optional[int] = None):
        self.data_root = data_root
        self.entries = sorted(
            e for e in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, e))
        )
        if max_images is not None and max_images < len(self.entries):
            self.entries = self.entries[:max_images]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        for entry_name in self.entries:
            entry_dir = os.path.join(self.data_root, entry_name)

            # ---------- Load RGB image ------------------------------------
            img_path = os.path.join(entry_dir, "frame.color.jpg")
            if not os.path.exists(img_path):
                print(f"  Warning: missing {img_path}, skipping.")
                continue
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"  Warning: could not read {img_path}, skipping.")
                continue

            # ---------- Load depth (Euclidean range) ----------------------
            depth_path = os.path.join(entry_dir, "frame.depth_meters.hdf5")
            if not os.path.exists(depth_path):
                print(f"  Warning: missing {depth_path}, skipping.")
                continue
            with h5py.File(depth_path, "r") as f:
                # Hypersim stores as 'dataset'
                key = list(f.keys())[0]
                gt_depth = np.array(f[key], dtype=np.float32)

            # ---------- Validity mask (not NaN, not inf, > 0) -------------
            valid_mask = np.isfinite(gt_depth) & (gt_depth > 0)

            # Replace NaN/inf with 0 for safe downstream processing
            gt_depth = np.where(valid_mask, gt_depth, 0.0).astype(np.float32)

            yield image_bgr, gt_depth, valid_mask, entry_name
