import argparse
import os
import random
import shutil
import sys
import zipfile

import pandas as pd
import requests

# Fix for importing from the same directory when run as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from download_helper import WebFile

# Increase download speed (as done in download.py)
zipfile.ZipExtFile.MIN_READ_SIZE = 2**20


def main():
    parser = argparse.ArgumentParser(description="Download Hypersim test set data.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/test_set_entries_names.csv",
        help="Path to test set metadata CSV",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of random samples to download",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hypersim",
        help="Root directory for downloaded data",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file {args.metadata} not found.")
        return

    df = pd.read_csv(args.metadata)

    # Select random subset
    if len(df) > args.num_samples:
        entries = len(df)
        random.seed(args.seed)
        df = df.sample(n=args.num_samples, random_state=args.seed)
        print(
            f"Selected {args.num_samples} random samples from {entries} total test set entries."
        )
    else:
        print(f"Downloading all {len(df)} available test set entries.")

    session = requests.Session()

    # Group by scene for efficiency
    grouped = df.groupby("scene_name")
    total_scenes = len(grouped)

    for i, (scene, scene_df) in enumerate(grouped):
        print(f"Progress: Scene {i + 1}/{total_scenes} ({len(scene_df)} entries)")
        download_scene_data(str(scene), scene_df, args.output_dir, session)


def download_scene_data(
    scene: str, scene_df: pd.DataFrame, target_root: str, session: requests.Session
):
    """
    Downloads all required files for all entries in a scene.
    Opens the scene ZIP file only once for efficiency.
    """
    url = f"https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{scene}.zip"

    try:
        wf = WebFile(url, session)
        with zipfile.ZipFile(wf) as z:
            for _, row in scene_df.iterrows():
                camera = str(row["camera_name"])
                frame = int(row["frame_id"])
                frame_padded = f"{frame:04d}"

                entry_id = f"{scene}_{camera}_{frame}"
                target_dir = os.path.join(target_root, entry_id)
                os.makedirs(target_dir, exist_ok=True)

                # Map from zip path to local filename
                files_to_download = {
                    f"{scene}/images/scene_{camera}_final_preview/frame.{frame_padded}.color.jpg": "frame.color.jpg",
                    f"{scene}/images/scene_{camera}_final_hdf5/frame.{frame_padded}.color.hdf5": "frame.color.hdf5",
                    f"{scene}/images/scene_{camera}_geometry_hdf5/frame.{frame_padded}.depth_meters.hdf5": "frame.depth_meters.hdf5",
                    f"{scene}/images/scene_{camera}_geometry_preview/frame.{frame_padded}.depth_meters.png": "frame.depth_meters.png",
                }

                for zip_path, local_name in files_to_download.items():
                    dest_path = os.path.join(target_dir, local_name)

                    if os.path.exists(dest_path):
                        continue

                    try:
                        with (
                            z.open(zip_path) as source,
                            open(dest_path, "wb") as target,
                        ):
                            shutil.copyfileobj(source, target)
                    except KeyError:
                        print(f"    Warning: {zip_path} not found in zip")
    except Exception as e:
        print(f"Error downloading scene {scene}: {e}")


if __name__ == "__main__":
    main()
