"""
handler.py — vascx-runpod
==========================
RunPod serverless worker for VascX artery/vein segmentation and fovea
localization. This is the pure-inference half of the fundus-vascx stack;
the Horizon MCP server handles preprocessing, MCP protocol, and response
formatting, then dispatches here for the GPU-bound forward pass only.

Expected input schema (job["input"])
-------------------------------------
{
    "task":     "av" | "fovea",          # required
    "image_id": str,                     # required — used in logs / response
    "rgb_b64":  str,                     # base64 PNG — preprocessed RGB crop
    "ce_b64":   str,                     # base64 PNG — preprocessed CE crop
}

Output schema (returned inside RunPod's {"output": ...} envelope)
-----------------------------------------------------------------
Task "av":
{
    "success":    true,
    "task":       "av",
    "image_id":   str,
    "av_raw_b64": str,   # base64-encoded uint8 flat array (row-major), dtype uint8
    "shape":      [H, W], # shape of av_raw, needed to reconstruct on the caller side
}

Task "fovea":
{
    "success":  true,
    "task":     "fovea",
    "image_id": str,
    "x":        float,   # mean_x from HeatmapRegressionEnsemble DataFrame output
    "y":        float,   # mean_y from HeatmapRegressionEnsemble DataFrame output
}

Error shape (both tasks):
{
    "success":  false,
    "error":    str,
    "image_id": str | null,
}

Design notes
------------
- Models are loaded ONCE at module level (global _runner). RunPod reuses the
  same worker process across jobs within a container lifetime, so this gives
  us free model caching at no extra code cost. The first job pays the cold-
  start cost (~3-5 s for two TorchScript loads); subsequent jobs are instant.

- predict_preprocessed takes a list of (rgb_path, ce_path) tuples as its
  first argument — not separate lists, not arrays. We decode the incoming
  base64 PNGs, save them as real PNG files inside a TemporaryDirectory, and
  pass [(rgb_path, ce_path)] as the paired_paths list. The temp directory is
  cleaned up automatically after each inference call.

- For AV segmentation, predict_preprocessed writes result PNGs to dest_path
  and returns None. The result PNG is read back from dest_path/rgb.png
  (stem matches the input rgb filename).

- For fovea localisation, predict_preprocessed returns a DataFrame with
  columns [mean_x, mean_y]. No dest_path is needed.

- num_workers=0 avoids multiprocessing inside the RunPod container, which
  can cause issues with forked processes in a serverless environment.

- av_raw is returned as a raw base64-encoded byte string of the flat uint8
  numpy array (plus shape metadata). The caller can reconstruct with:
      np.frombuffer(base64.b64decode(av_raw_b64), dtype=np.uint8).reshape(shape)
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import tempfile
import torch
from pathlib import Path

import runpod

logging.basicConfig(
    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("vascx-worker")

# Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEIGHTS_DIR   = Path(os.environ.get("WEIGHTS_DIR", "/runpod-volume"))
AV_WEIGHTS    = WEIGHTS_DIR / "av_july24.pt"
FOVEA_WEIGHTS = WEIGHTS_DIR / "fovea_july24.pt"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class _Runner:
    """Holds both loaded TorchScript ensembles."""
    __slots__ = ("seg", "hm")

    def __init__(self):
        import cv2
        cv2.setNumThreads(1)

        from rtnls_inference import SegmentationEnsemble, HeatmapRegressionEnsemble

        logger.info(f"Loading AV weights from {AV_WEIGHTS} ...")
        self.seg = SegmentationEnsemble.from_torchscript(str(AV_WEIGHTS)).to(device)
        logger.info("AV weights loaded.")

        logger.info(f"Loading fovea weights from {FOVEA_WEIGHTS} ...")
        self.hm = HeatmapRegressionEnsemble.from_torchscript(str(FOVEA_WEIGHTS)).to(device)
        logger.info("Fovea weights loaded.")


def _load_runner() -> _Runner:
    """Validate weight files exist then load models. Raises on missing files."""
    import zipfile
    for p in (AV_WEIGHTS, FOVEA_WEIGHTS):
        if not p.exists():
            raise FileNotFoundError(
                f"Weight file not found: {p}\n"
                "Upload the .pt files to the attached RunPod network volume."
            )
        try:
            with zipfile.ZipFile(p):
                pass
        except zipfile.BadZipFile:
            raise RuntimeError(
                f"{p} is a Git LFS pointer, not a real weight file. "
                "Run `git lfs pull` before building the image."
            )
    return _Runner()


logger.info("Initialising VascX models ...")
try:
    _runner = _load_runner()
    logger.info("VascX models ready.")
except Exception as _e:
    logger.error(f"Model initialisation failed: {_e}", exc_info=True)
    raise


# ---------------------------------------------------------------------------
# Input decoding helpers
# ---------------------------------------------------------------------------

def _decode_image(b64: str):
    """Decode a base64 PNG into a uint8 numpy HxWx3 array."""
    import io
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _validate_input(job_input: dict) -> tuple[str, str, str, str]:
    """
    Pull and validate required fields. Returns (task, image_id, rgb_b64, ce_b64).
    Raises ValueError with a descriptive message on any missing / bad field.
    """
    task = job_input.get("task")
    if task not in ("av", "fovea"):
        raise ValueError(f"'task' must be 'av' or 'fovea', got: {task!r}")

    image_id = job_input.get("image_id")
    if not image_id:
        raise ValueError("'image_id' is required and must be a non-empty string.")

    rgb_b64 = job_input.get("rgb_b64")
    ce_b64  = job_input.get("ce_b64")
    if not rgb_b64:
        raise ValueError("'rgb_b64' (base64 preprocessed RGB crop) is required.")
    if not ce_b64:
        raise ValueError("'ce_b64' (base64 preprocessed CE crop) is required.")

    return task, image_id, rgb_b64, ce_b64


# ---------------------------------------------------------------------------
# Shared helper — save decoded arrays to disk and return paired_paths list
# ---------------------------------------------------------------------------

def _save_images(rgb_arr, ce_arr, tmp: Path) -> list[tuple[Path, Path]]:
    """
    Save numpy arrays as PNGs and return a paired_paths list as expected by
    predict_preprocessed: [(rgb_path, ce_path)].
    """
    from PIL import Image

    rgb_path = tmp / "rgb.png"
    ce_path  = tmp / "ce.png"
    Image.fromarray(rgb_arr).save(rgb_path)
    Image.fromarray(ce_arr).save(ce_path)
    return [(rgb_path, ce_path)]


# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------

def _run_av(rgb_arr, ce_arr, image_id: str) -> dict:
    """
    Run AV segmentation and return the raw prediction dict.

    predict_preprocessed writes the result PNG to dest_path/<rgb_stem>.png
    and returns None, so we read the result back from disk.
    """
    import numpy as np
    from PIL import Image

    logger.info(f"[{image_id}] Running AV segmentation ...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        paired_paths = _save_images(rgb_arr, ce_arr, tmp)
        dest_path = tmp / "av"
        dest_path.mkdir()

        # Returns None — result is written to dest_path/rgb.png
        _runner.seg.predict_preprocessed(
            paired_paths, dest_path=dest_path, num_workers=0
        )

        # Read result back from disk; stem matches input rgb filename
        result_path = dest_path / "rgb.png"
        if not result_path.exists():
            raise FileNotFoundError(
                f"Expected AV result at {result_path} but file was not written. "
                f"Contents of dest_path: {list(dest_path.iterdir())}"
            )
        av_raw = np.array(Image.open(result_path))

    av_raw = av_raw.astype(np.uint8)

    # Zero-mask pixels that are pure black in the RGB (outside the fundus disc)
    black_mask = rgb_arr.max(axis=2) <= 10
    av_raw[black_mask] = 0

    logger.info(
        f"[{image_id}] AV done — shape {av_raw.shape}, "
        f"non-zero px: {int((av_raw > 0).sum())}"
    )

    # Serialise as raw bytes; caller reconstructs with:
    # np.frombuffer(base64.b64decode(av_raw_b64), dtype=np.uint8).reshape(shape)
    av_raw_b64 = base64.b64encode(av_raw.tobytes()).decode("ascii")

    return {
        "success":    True,
        "task":       "av",
        "image_id":   image_id,
        "av_raw_b64": av_raw_b64,
        "shape":      list(av_raw.shape),
    }


def _run_fovea(rgb_arr, ce_arr, image_id: str) -> dict:
    """
    Run fovea localisation and return (x, y) in preprocessed image space.

    HeatmapRegressionEnsemble.predict_preprocessed returns a DataFrame with
    columns [mean_x, mean_y] (one row per image). No dest_path needed.
    """
    logger.info(f"[{image_id}] Localising fovea ...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        paired_paths = _save_images(rgb_arr, ce_arr, tmp)

        df = _runner.hm.predict_preprocessed(paired_paths, num_workers=0)

    x = float(df.iloc[0, 0])  # mean_x
    y = float(df.iloc[0, 1])  # mean_y
    logger.info(f"[{image_id}] Fovea at x={x:.1f}, y={y:.1f}")

    return {
        "success":  True,
        "task":     "fovea",
        "image_id": image_id,
        "x":        x,
        "y":        y,
    }


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """
    RunPod synchronous handler.

    RunPod wraps the returned dict as {"output": <return value>} automatically.
    We catch and return structured errors so the caller always gets a
    parseable payload rather than an SDK-level FAILED status.
    """
    job_input = job.get("input", {})
    image_id  = job_input.get("image_id", "<unknown>")

    try:
        task, image_id, rgb_b64, ce_b64 = _validate_input(job_input)
        rgb_arr = _decode_image(rgb_b64)
        ce_arr  = _decode_image(ce_b64)
    except (ValueError, Exception) as exc:
        logger.error(f"[{image_id}] Input error: {exc}")
        return {"success": False, "error": str(exc), "image_id": image_id}

    try:
        if task == "av":
            return _run_av(rgb_arr, ce_arr, image_id)
        else:
            return _run_fovea(rgb_arr, ce_arr, image_id)
    except Exception as exc:
        logger.error(f"[{image_id}] Inference error: {exc}", exc_info=True)
        return {"success": False, "error": str(exc), "image_id": image_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
