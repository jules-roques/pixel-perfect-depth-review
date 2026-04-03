# Pixel-Perfect Depth Review (ppdr)

This project provides wrappers and tools for evaluating depth estimation models, specifically focusing on Pixel-Perfect Depth (PPD).

## Installation with uv

We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable package management. To install it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, to install all the dependencies and the project in editable mode:
```bash
uv sync
```

## Download data and models

```bash
uv run scripts/download/download_test_data.py
uv run scripts/download/download_models.py
```

## Third-party Code Attribution

This project includes code from the following third-party source:
- **[Pixel-Perfect-Depth](https://github.com/gangweix/pixel-perfect-depth)** (Apache 2.0)
  - Vendored subset located in [src/ppdr/vendor/ppd/](src/ppdr/vendor/ppd/)
  - Original license: [LICENSE](src/ppdr/vendor/ppd/LICENSE)
  - Detailed credits and modifications: [README.md](src/ppdr/vendor/ppd/README.md)
