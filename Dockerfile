# =============================================================================
# vascx-runpod — RunPod serverless GPU worker
# =============================================================================
# Runs VascX artery/vein segmentation and fovea localisation on GPU.
# This image is the inference-only half of the fundus-vascx stack; the
# Horizon MCP server handles preprocessing and MCP protocol, then calls
# this worker via the RunPod Serverless endpoint API.
#
# Base image: runpod/pytorch ships CUDA + cuDNN + a GPU-built torch/torchvision
# so we never pull a CPU-only torch from PyPI.
#
# Tested base: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Pin this in CI; "latest" drifts.
# =============================================================================

ARG PYTORCH_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
FROM ${PYTORCH_IMAGE}

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
# libglib2.0-0  — required by OpenCV headless (GLib dependency)
# libgomp1      — OpenMP, used by torch CPU kernels and joblib in fundusprep
# No X11 libs — we explicitly want the headless OpenCV build.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies — install order is intentional (see requirements.txt)
# ---------------------------------------------------------------------------

COPY requirements.txt .

# retinalysis-inference has tight transitive pins that fight the broader stack;
# install it without deps so we own the resolution.
RUN pip install --no-cache-dir --no-deps retinalysis-inference

# Install everything else, then force-reinstall headless OpenCV last.
# This unconditionally evicts the GUI variant (opencv-python) regardless of
# whatever retinalysis-fundusprep pulled in transitively, without needing
# any pre-install ordering or post-install checks.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall "opencv-python-headless>=4.0,<5.0"

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY handler.py .

# ---------------------------------------------------------------------------
# Model weights — loaded from a RunPod network volume at runtime.
#
# Weights are NOT baked into the image. Instead, attach a network volume
# in your RunPod serverless template and upload artery-vein.pt and
# vascx-fovea.pt to it. Set the WEIGHTS_DIR env var in the template to
# match the volume mount path (e.g. /runpod-volume).
#
# To upload weights to the volume:
#   1. Create a network volume in the RunPod console.
#   2. Spin up a temporary CPU pod with the volume attached.
#   3. scp or wget your .pt files onto the pod at the mount path.
#   4. Terminate the pod — the files persist on the volume.
#   5. Attach the same volume to your serverless template.
# ---------------------------------------------------------------------------

# Default matches the standard RunPod network volume mount path.
# Override in the serverless template env vars if your volume is mounted elsewhere.
ENV WEIGHTS_DIR=/runpod-volume

# ---------------------------------------------------------------------------
# Non-root user — good hygiene; RunPod doesn't require root.
# ---------------------------------------------------------------------------
RUN useradd --no-create-home --shell /bin/false worker \
    && chown -R worker:worker /app
USER worker

# ---------------------------------------------------------------------------
# Entry point
# RunPod executes CMD to start the worker. handler.py calls
# runpod.serverless.start() which blocks and polls for jobs.
# ---------------------------------------------------------------------------
CMD ["python", "-u", "handler.py"]

# =============================================================================
# Build instructions
# =============================================================================
#
# 1. Pull real weights (do this once per machine):
#       git lfs install && git lfs pull
#
# 2. Build:
#       docker build -t your-registry/vascx-runpod:latest .
#
# 3. Push to a registry RunPod can pull from (Docker Hub, GHCR, etc.):
#       docker push your-registry/vascx-runpod:latest
#
# 4. Create a RunPod Serverless Endpoint:
#       - Container image: your-registry/vascx-runpod:latest
#       - GPU type: any NVIDIA with ≥8 GB VRAM (RTX 3090, A5000, etc.)
#       - Min workers: 1 (eliminates cold-start model-load latency for
#         production; set to 0 for dev/low-traffic to save cost)
#       - No environment variables required (weights are baked in).
#
# 5. Local test (CPU, no GPU required):
#       docker run --rm -p 8000:8000 your-registry/vascx-runpod:latest \
#           python handler.py --rp_serve_api
#       # then POST to http://localhost:8000/runsync
# =============================================================================
