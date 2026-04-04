import argparse
import os

import torch

from ppdr.models import PPD, DAv2, DAv2Cleaned
from ppdr.utils.dataset import HypersimDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute depth maps for visualization."
    )
    parser.add_argument(
        "--entries_names",
        type=str,
        nargs="+",
        required=True,
        help="Names of the entries to precompute.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/precomputed",
        help="Path to the output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HypersimDataset()
    dav2 = DAv2(device=device)
    dav2_cleaned = DAv2Cleaned(device=device)
    ppd = PPD(device=device)

    for entry_name in args.entries_names:
        entry = dataset.get_entry_by_name(entry_name)
        out = f"{args.output_path}/{entry_name}"
        os.makedirs(out, exist_ok=True)
        image = entry["image"].unsqueeze(0).to(device)

        disp, _ = dav2.predict(image)
        torch.save(disp.cpu(), f"{out}/disp_dav2.pt")

        disp_cleaned, _ = dav2_cleaned.predict(image)
        torch.save(disp_cleaned.cpu(), f"{out}/disp_dav2_clean.pt")

        depth, _ = ppd.predict(image)
        torch.save(depth.cpu(), f"{out}/depth_ppd.pt")


if __name__ == "__main__":
    main()
