# Vendored: Pixel-Perfect-Depth

This directory contains a vendored subset of the [Pixel-Perfect-Depth](https://github.com/gangweix/pixel-perfect-depth) repository.

## Origin
- **Source Repository**: `gangweix/pixel-perfect-depth`
- **Original License**: Apache License 2.0 (see [LICENSE](./LICENSE))

## Purpose
This code is vendored to provide robust inference for:
- **Pixel-Perfect-Depth (PPD)**
- **Depth-Anything-V2** (via DPT and DINOv2 layers)
- **MoGe** geometry utilities

## Modifications
- Removed training, dataset loading, and evaluation scripts to keep only the inference-required subset.
- Updated internal imports from `ppd.*` to `ppdr.vendor.ppd.*` for compatibility with the `ppdr` package structure.
- Removed unused dependencies (e.g., `omegaconf`).
