# CORAL-v3 training image
# Build: docker build -t <dockerhub-user>/coral-v3:latest .
# Push:  docker push <dockerhub-user>/coral-v3:latest
# Use on Vast.ai: specify this image when provisioning.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (git for repo clone, build-essential for any source builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install from requirements.txt ahead of cloning the repo
# so Docker can cache this layer across repo changes.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install flash-attn (A100-only arch for faster build; remove ARCHS to build all)
RUN FLASH_ATTN_CUDA_ARCHS=80 pip install --no-cache-dir "flash-attn>=2.7,<3.0" --no-build-isolation

# Working directory
WORKDIR /workspace

# On Vast.ai, the user clones CORAL-v3 into /workspace/CORAL-v3 and runs from there.
