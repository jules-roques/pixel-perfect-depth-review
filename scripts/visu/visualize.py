"""
Generates visuals from precomputed depths.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from ppdr.utils.dataset import HypersimDataset
from ppdr.utils.geometry import (
    create_valid_depth_mask,
    depth_to_point_cloud,
    edge_mask,
    recover_metric_depth_from_disparity,
    recover_metric_depth_from_log,
)
from ppdr.utils.transform import image_tensor2array


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize point clouds.")
    parser.add_argument("--entry_name", type=str, required=True, help="Entry name.")
    parser.add_argument(
        "--out_dir", type=str, default="results/visuals", help="Output directory."
    )
    return parser.parse_args()


def save_cloud(pts: np.ndarray, colors: np.ndarray, out_path: str) -> None:
    plotter = pv.Plotter(off_screen=False)
    pdata = pv.PolyData(pts)
    pdata["colors"] = (colors * 255).astype(np.uint8)
    plotter.add_mesh(pdata, scalars="colors", rgb=True, point_size=2)
    plotter.show(auto_close=False)
    plotter.save_graphic(out_path)
    plotter.close()


def main():

    args = parse_args()

    out_dir = f"{args.out_dir}/{args.entry_name}"
    os.makedirs(out_dir, exist_ok=True)

    dataset = HypersimDataset()
    entry = dataset.get_entry_by_name(args.entry_name)
    image = entry["image"]
    gt_depth = entry["depth"].unsqueeze(0)
    gt_valid_mask = entry["valid_mask"]
    m_cam_from_uv = entry["ndc_to_cam"]

    # image, depth and mask
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_tensor2array(image))
    axs[0].axis("off")
    axs[0].set_title("Image")
    axs[1].imshow(gt_depth.squeeze(0), cmap="magma")
    axs[1].axis("off")
    axs[1].set_title("GT Depth")
    axs[2].imshow(edge_mask(image.unsqueeze(0), 0.2, 0.5, 1).squeeze(0))
    axs[2].axis("off")
    axs[2].set_title("Edge Mask (using RGB)")
    fig.tight_layout(pad=1)
    fig.savefig(f"{out_dir}/image_depth_mask.png")
    plt.show()

    ppd_depth = torch.load(f"results/precomputed/{args.entry_name}/depth_ppd.pt")
    disp_dav2 = torch.load(f"results/precomputed/{args.entry_name}/disp_dav2.pt")
    disp_dav2_clean = torch.load(
        f"results/precomputed/{args.entry_name}/disp_dav2_clean.pt"
    )

    valid_mask_ppd = create_valid_depth_mask(ppd_depth)
    valid_mask_dav2 = create_valid_depth_mask(disp_dav2)
    valid_mask_dav2_clean = create_valid_depth_mask(disp_dav2_clean)

    ppd_depth_aligned = recover_metric_depth_from_log(
        ppd_depth, gt_depth, valid_mask_ppd
    )
    disp_dav2_aligned = recover_metric_depth_from_disparity(
        disp_dav2, gt_depth, valid_mask_dav2
    )
    disp_dav2_clean_aligned = recover_metric_depth_from_disparity(
        disp_dav2_clean, gt_depth, valid_mask_dav2_clean
    )

    ppd_pts = depth_to_point_cloud(
        ppd_depth_aligned.squeeze(0), m_cam_from_uv, valid_mask_ppd.squeeze(0)
    )
    dav2_pts = depth_to_point_cloud(
        disp_dav2_aligned.squeeze(0), m_cam_from_uv, valid_mask_dav2.squeeze(0)
    )
    dav2_clean_pts = depth_to_point_cloud(
        disp_dav2_clean_aligned.squeeze(0),
        m_cam_from_uv,
        valid_mask_dav2_clean.squeeze(0),
    )
    gt_pts = depth_to_point_cloud(
        gt_depth.squeeze(0),
        m_cam_from_uv,
        gt_valid_mask.squeeze(0),
    )

    save_cloud(
        ppd_pts.numpy(),
        image_tensor2array(image)[valid_mask_ppd.squeeze(0).cpu().numpy()],
        f"{out_dir}/ppd_cloud.svg",
    )
    save_cloud(
        dav2_pts.numpy(),
        image_tensor2array(image)[valid_mask_dav2.squeeze(0).cpu().numpy()],
        f"{out_dir}/dav2_cloud.svg",
    )
    save_cloud(
        dav2_clean_pts.numpy(),
        image_tensor2array(image)[valid_mask_dav2_clean.squeeze(0).cpu().numpy()],
        f"{out_dir}/dav2_clean_cloud.svg",
    )

    save_cloud(
        gt_pts.numpy(),
        image_tensor2array(image)[gt_valid_mask.squeeze(0).cpu().numpy()],
        f"{out_dir}/gt_cloud.svg",
    )


if __name__ == "__main__":
    main()
