"""
Hypersim HDF5 + image data loader for the benchmark.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


class HypersimReader:
    """
    Iterate over downloaded Hypersim test-set folders.

    Each folder contains:
        - ``frame.color.hdf5``          (high dynamic range photometric RGB image)
        - ``frame.depth_meters.hdf5``  (Euclidean range depth)

    Yields ``(entry_name, image_hdr, distances_from_camera, ndc_to_cam)``
    tuples.
    """

    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.ndc_to_cam_map = self._load_ndc_to_cam_map()
        self.entries_names = self._get_entries_names()
        self.entry_name_to_index = {
            name: i for i, name in enumerate(self.entries_names)
        }

    def get_entry_index_from_name(self, name: str) -> int:
        return self.entry_name_to_index[name]

    def get_number_entries(self) -> int:
        return len(self.entries_names)

    def get_entry_by_index(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx < 0 or idx >= len(self.entries_names):
            raise IndexError(f"Index {idx} is out of bounds.")
        return self._load_entry(self.entries_names[idx])

    def get_entry_by_name(
        self, entry_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if entry_name not in self.entries_names:
            raise ValueError(f"Entry name {entry_name} not found in the dataset.")
        return self._load_entry(entry_name)

    def _get_entries_names(self) -> list[str]:
        return sorted([f.name for f in self.data_root.iterdir() if f.is_dir()])

    def _load_entry(self, entry_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_rgb = self._load_image(entry_name)
        distances = self._load_distances_to_camera(entry_name)
        ndc_to_cam = self._get_ndc_to_cam(entry_name)
        return image_rgb, distances, ndc_to_cam

    def _load_ndc_to_cam_map(self) -> dict[str, np.ndarray]:
        csv_path = self.data_root.parent / "uv_to_cam.csv"
        df = pd.read_csv(csv_path)

        ndc_to_cam_map = {}
        for _, row in df.iterrows():
            ndc_to_cam_map[row["scene_name"]] = np.array(
                [
                    [row["uv_to_cam_00"], row["uv_to_cam_01"], row["uv_to_cam_02"]],
                    [row["uv_to_cam_10"], row["uv_to_cam_11"], row["uv_to_cam_12"]],
                    [row["uv_to_cam_20"], row["uv_to_cam_21"], row["uv_to_cam_22"]],
                ],
                dtype=np.float32,
            )
        return ndc_to_cam_map

    def _load_distances_to_camera(self, entry_name: str) -> np.ndarray:

        depth_path = self.data_root / entry_name / "frame.depth_meters.hdf5"
        if not depth_path.exists():
            raise FileNotFoundError(f"Distance file not found: {depth_path}")
        with h5py.File(str(depth_path), "r") as f:
            key = list(f.keys())[0]
            distances = np.array(f[key], dtype=np.float32)

        return distances

    def _load_image(self, entry_name: str) -> np.ndarray:
        img_path = self.data_root / entry_name / "frame.color.hdf5"
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        with h5py.File(str(img_path), "r") as f:
            key = list(f.keys())[0]
            image_hdr = np.array(f[key], dtype=np.float32)
        return image_hdr

    def _get_ndc_to_cam(self, entry_name: str) -> np.ndarray:
        scene_name = self._get_scene_name(entry_name)
        ndc_to_cam = self.ndc_to_cam_map.get(scene_name)
        if ndc_to_cam is None:
            raise ValueError(f"Missing intrinsics metadata for {scene_name}")
        return ndc_to_cam

    def _get_scene_name(self, entry_name: str) -> str:
        return "_".join(entry_name.split("_")[:3])
