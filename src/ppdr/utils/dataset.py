import torch
from torch.utils.data import Dataset

from ppdr.utils.geometry import create_valid_depth_mask, distances_from_camera_to_depth
from ppdr.utils.reader import HypersimReader
from ppdr.utils.transform import image_array2tensor, linear_to_rgb


class HypersimDataset(Dataset):
    def __init__(self, hypersim_reader: HypersimReader):
        self.reader = hypersim_reader

    def __len__(self) -> int:
        return self.reader.get_number_entries()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:

        image_hdr, distances_from_camera, ndc_to_cam = self.reader.get_entry_by_index(
            index
        )

        image_rgb = linear_to_rgb(image_hdr)
        image_tensor = image_array2tensor(image_rgb)
        distances_tensor = torch.from_numpy(distances_from_camera)
        ndc_to_cam_tensor = torch.from_numpy(ndc_to_cam)
        depth_tensor = distances_from_camera_to_depth(
            distances_tensor, ndc_to_cam_tensor
        )

        valid_mask = create_valid_depth_mask(depth_tensor)

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask,
            "ndc_to_cam": ndc_to_cam_tensor,
        }

    def get_entry_by_name(self, name: str) -> dict[str, torch.Tensor]:
        index = self.reader.get_entry_index_from_name(name)
        return self.__getitem__(index)
