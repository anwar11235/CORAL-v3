# CORAL v3 — COrtical Reasoning via Abstraction Layers

Neural architecture for complex reasoning tasks built on the HRM framework, adding
precision-weighted predictive coding, recognition-gated crystallization, and sparse
columnar routing under a variational free energy objective.

## GPU Provisioning

For Vast.ai or similar GPU rentals, use the prebuilt Docker image:

    <dockerhub-user>/coral-v3:latest

This image has torch 2.6 + CUDA 12.4 + flash-attn preinstalled, reducing
provision time from ~45 minutes (clone, install deps, build flash-attn) to ~2 minutes.

Build and push the image yourself:

    docker build -t <dockerhub-user>/coral-v3:latest .
    docker push <dockerhub-user>/coral-v3:latest

See `Dockerfile` for details.

## Quick Start

    pip install -e .
    python scripts/train.py --config-name base

## Dataset

See `coral/data/README.md` for instructions on building the Sudoku-Extreme-1K dataset.
