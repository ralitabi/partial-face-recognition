"""
Predict identity from a partial face image.

Usage:
    python Model/predict.py <image_path>
    python Model/predict.py          # auto-picks first test image
"""

import os, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_utils import load_model_from_checkpoint, eval_tf

import torch
from PIL import Image

BEST_PATH   = ROOT / "best_model.pt"
MODEL_PATH  = ROOT / "partial_face_model.pt"
DATASET_DIR = ROOT / "partial_face_dataset"
TOP_K       = 5

_model        = None
_idx_to_class = None
_lookup       = None   # (identity, transformation) -> filename


def _ensure_loaded():
    global _model, _idx_to_class, _lookup
    if _model is not None:
        return
    path = BEST_PATH if BEST_PATH.exists() else MODEL_PATH
    if not path.exists():
        sys.exit("[ERROR] No model found. Run: python Model/pretrained_model.py")
    _model, _idx_to_class = load_model_from_checkpoint(str(path))
    df      = pd.read_csv(DATASET_DIR / "metadata.csv")
    _lookup = df.set_index(["identity", "transformation"])["filename"].to_dict()


def predict(img_path: str) -> list:
    _ensure_loaded()
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read: {img_path}")
        return []

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor  = eval_tf(pil_img).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(_model(tensor)[0], dim=0).numpy()

    top_idx = probs.argsort()[-TOP_K:][::-1]
    results = [(_idx_to_class[i], float(probs[i])) for i in top_idx]

    print(f"\nFile : {img_path}")
    print(f"{'Rank':<5} {'Identity':<40} Confidence")
    print("-" * 60)
    for rank, (name, conf) in enumerate(results, 1):
        print(f"  {rank}    {name:<40} {conf*100:6.2f}%  {'#'*int(conf*25)}")

    best     = results[0][0]
    ref_file = _lookup.get((best, "original"))
    ref_img  = cv2.imread(str(DATASET_DIR / ref_file)) if ref_file else None

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image"); axes[0].axis("off")
    if ref_img is not None:
        axes[1].imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Best Match\n{best[:30]}")
    else:
        axes[1].set_title("No reference found")
    axes[1].axis("off")
    plt.suptitle(f"Top-1: {best} ({results[0][1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=120)
    plt.close()
    print("Saved: prediction_result.png")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        _ensure_loaded()
        df   = pd.read_csv(DATASET_DIR / "metadata.csv")
        rows = df[df["split"] == "test"] if "split" in df.columns else df
        if rows.empty:
            print("Pass an image path as argument.")
        else:
            predict(str(DATASET_DIR / rows.iloc[0]["filename"]))
