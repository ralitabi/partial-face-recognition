"""Shared utilities for all model scripts and the Streamlit app."""

import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=8),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FaceDataset(Dataset):
    def __init__(self, df, dataset_dir, transform, class_to_idx):
        self.df           = df.reset_index(drop=True)
        self.dataset_dir  = Path(dataset_dir)
        self.transform    = transform
        self.class_to_idx = class_to_idx
        self._missing     = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = self.dataset_dir / row["filename"]
        label = self.class_to_idx.get(row["identity"])
        if label is None:
            raise KeyError(f"Identity '{row['identity']}' not in class_to_idx. "
                           f"Re-run training after adding new data.")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            self._missing += 1
            if self._missing <= 5:
                warnings.warn(f"[FaceDataset] Cannot open {path} — substituting blank image.")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        return self.transform(img), label


def validate_metadata(df, class_to_idx, split_name=""):
    """Raise early with a clear message if metadata is malformed."""
    required_cols = {"identity", "filename"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"metadata.csv missing columns: {missing_cols}")
    unknown = set(df["identity"].unique()) - set(class_to_idx)
    if unknown:
        raise ValueError(
            f"{len(unknown)} identities in {split_name} not found in class_to_idx. "
            f"Examples: {list(unknown)[:3]}"
        )


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build EfficientNet-B0 with custom classification head."""
    weights  = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.efficientnet_b0(weights=weights)
    in_feat  = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_feat, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return backbone


def load_checkpoint(path, device="cpu") -> dict:
    """Load a .pt checkpoint and verify it has required keys."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for key in ("model_state", "num_classes", "class_to_idx", "idx_to_class"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint {path} missing key '{key}'. "
                             f"Re-run pretrained_model.py to rebuild it.")
    return ckpt


def load_model_from_checkpoint(path, device="cpu"):
    """Return (model_eval, idx_to_class) ready for inference."""
    ckpt  = load_checkpoint(path, device)
    model = build_model(ckpt["num_classes"], pretrained=True)
    # Load full state — if only head weights are stored, load strictly=False
    result = model.load_state_dict(ckpt["model_state"], strict=False)
    if result.unexpected_keys:
        warnings.warn(f"Unexpected keys in checkpoint: {result.unexpected_keys[:3]}")
    model.to(device).eval()
    return model, ckpt["idx_to_class"]


def accuracy(preds: list, labels: list) -> float:
    return float(np.mean(np.array(preds) == np.array(labels)))
