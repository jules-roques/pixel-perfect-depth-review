import os
import subprocess
import sys

CHECKPOINTS_DIR = "checkpoints"

MODELS = {
    "ppd.pth": {
        "url": "https://huggingface.co/gangweix/Pixel-Perfect-Depth/resolve/main/ppd.pth",
        "label": "PPD (DA2)",
    },
    "depth_anything_v2_vitl.pth": {
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
        "label": "Depth Anything V2 Large",
    },
    "moge2.pt": {
        "url": "https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt?download=true",
        "label": "MoGe2",
    },
}


def main():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    print(f"Downloading checkpoints to {CHECKPOINTS_DIR}...")

    for filename, model in MODELS.items():
        dest = os.path.join(CHECKPOINTS_DIR, filename)
        print(f"Downloading {model['label']}...")
        download_file(model["url"], dest)

    print(f"Done! Checkpoints are in {CHECKPOINTS_DIR}/")


def download_file(url: str, dest: str):
    """Download a file using wget."""
    result = subprocess.run(
        ["wget", "-O", dest, url],
        check=False,
    )
    if result.returncode != 0:
        print(f"Error: failed to download {url}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
