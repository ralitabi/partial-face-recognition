"""
Deep Full-Model Training
========================
Fine-tunes the ENTIRE EfficientNet-B0 (backbone + head) on all 21,266 images.

Why this beats head-only training
-----------------------------------
Head-only (night_train.py) caps at ~88-89% because the backbone features
were extracted by a generic ImageNet backbone.  Training the backbone ON face
data lets it learn face-specific features → higher accuracy, F1, and confidence.

Why previous backbone attempts failed (17%)
--------------------------------------------
The old code compared validation accuracy against 0.0, so even a badly
degraded 17% model was saved as "best".  This script compares against the
CURRENT best_model.pt accuracy and only overwrites if genuinely improved.

Strategy
---------
Phase A – Head warmup  (5 epochs, backbone fully frozen)
    Get head adapted to current backbone before unlocking anything.

Phase B – Staged unfreeze  (backbone unlocked one block at a time)
    Unfreeze last 1 block → train until stable.
    Unfreeze last 2 blocks → train until stable.
    Unfreeze last 3 blocks → train until stable.
    Unfreeze full backbone → train until stable.

    At each stage, backbone LR = 1/10 of head LR.
    BatchNorm running stats kept frozen (only affine params update).
    Gradient clipping prevents instability.

Usage
------
    python deep_train.py
"""

import sys, gc, time, json, datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_utils import (
    IMG_SIZE, FaceDataset, build_model, load_checkpoint,
    eval_tf, train_tf, accuracy, validate_metadata
)

DATASET_DIR = ROOT / "partial_face_dataset"
BEST_PATH   = ROOT / "best_model.pt"
MODEL_PATH  = ROOT / "partial_face_model.pt"
RESULTS_DIR = ROOT / "results"
HIST_PATH   = RESULTS_DIR / "training_history.json"

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 16      # full backbone + gradient on CPU needs small batch
HEAD_LR       = 5e-4    # head learning rate
BACKBONE_LR   = 5e-5    # backbone LR = 1/10 of head (differential rates)
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.08
GRAD_CLIP     = 1.0     # max gradient norm
PATIENCE      = 6       # early-stopping patience per phase

# Epochs per phase
EPOCHS_WARMUP = 5       # Phase A: head warmup (backbone frozen)
EPOCHS_STAGE  = 8       # Phase B: epochs per unfreeze stage

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}", flush=True)

# ── Data ──────────────────────────────────────────────────────────────────────
meta = pd.read_csv(DATASET_DIR / "metadata.csv")
if "split" in meta.columns:
    train_df = meta[meta["split"] == "train"].reset_index(drop=True)
    val_df   = meta[meta["split"] == "val"].reset_index(drop=True)
    test_df  = meta[meta["split"] == "test"].reset_index(drop=True)
else:
    train_df, tmp   = train_test_split(meta, test_size=0.30, stratify=meta["identity"], random_state=SEED)
    val_df, test_df = train_test_split(tmp,  test_size=0.50, stratify=tmp["identity"],  random_state=SEED)

classes      = sorted(meta["identity"].unique().tolist())
num_classes  = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

validate_metadata(train_df, class_to_idx, "train")
validate_metadata(val_df,   class_to_idx, "val")
validate_metadata(test_df,  class_to_idx, "test")

print(f"Classes : {num_classes}", flush=True)
print(f"Train   : {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}", flush=True)

y_train      = train_df["identity"].map(class_to_idx).values
weights_arr  = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
class_weight = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(FaceDataset(train_df, DATASET_DIR, train_tf, class_to_idx),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(FaceDataset(val_df,   DATASET_DIR, eval_tf,  class_to_idx),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(FaceDataset(test_df,  DATASET_DIR, eval_tf,  class_to_idx),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_model(num_classes, pretrained=True).to(DEVICE)

# Load best checkpoint (head + backbone weights)
if BEST_PATH.exists():
    ckpt = load_checkpoint(BEST_PATH, DEVICE)
    if ckpt.get("num_classes") == num_classes:
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded best_model.pt checkpoint", flush=True)

# ── Get current best accuracy (the bar we must beat) ─────────────────────────
def current_best_acc():
    if not HIST_PATH.exists(): return 0.0
    h = json.loads(HIST_PATH.read_text())
    return max((r["test_acc"] for r in h), default=0.0)

overall_best_acc = current_best_acc()
print(f"Current best accuracy to beat: {overall_best_acc:.2f}%\n", flush=True)

criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=LABEL_SMOOTH)

# ── Training utilities ────────────────────────────────────────────────────────
def freeze_bn_running_stats(model):
    """Keep BN affine params trainable but freeze running mean/var."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()          # don't update running stats
            m.weight.requires_grad_(True)   # keep gamma trainable
            m.bias.requires_grad_(True)     # keep beta trainable

def set_backbone_blocks(model, num_unfrozen):
    """Freeze all backbone, then unfreeze the last num_unfrozen blocks."""
    for param in model.features.parameters():
        param.requires_grad = False
    if num_unfrozen > 0:
        blocks = list(model.features.children())
        for block in blocks[-num_unfrozen:]:
            for param in block.parameters():
                param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

def make_optimizer(model, head_lr, backbone_lr):
    """Differential LR: backbone gets backbone_lr, head gets head_lr."""
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "classifier" in n and p.requires_grad]
    return optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ], weight_decay=WEIGHT_DECAY)

def run_epoch(loader, train=True, optimizer=None):
    if train:
        model.train()
        freeze_bn_running_stats(model)   # always freeze BN running stats
    else:
        model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(imgs)
            if train and (batch_idx + 1) % 50 == 0:
                print(f"    batch {batch_idx+1}/{len(loader)} | "
                      f"loss={total_loss/total:.4f} | "
                      f"acc={correct/total*100:.1f}%", flush=True)
    return total_loss / total, correct / total

def train_phase(phase_name, epochs, num_backbone_blocks, head_lr, backbone_lr):
    """Run a training phase with the given unfreeze level."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {phase_name}", flush=True)
    print(f"  Backbone blocks unfrozen: {num_backbone_blocks}  "
          f"| Head LR: {head_lr}  | Backbone LR: {backbone_lr}", flush=True)
    print(f"{'='*60}", flush=True)

    set_backbone_blocks(model, num_backbone_blocks)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_train:,} / {n_total:,} ({n_train/n_total*100:.1f}%)", flush=True)

    optimizer = make_optimizer(model, head_lr, backbone_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7)

    best_val   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{epochs} — training…", flush=True)
        tr_loss, tr_acc = run_epoch(train_loader, train=True,  optimizer=optimizer)
        print(f"Epoch {epoch}/{epochs} — validating…", flush=True)
        va_loss, va_acc = run_epoch(val_loader,   train=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train {tr_acc*100:.2f}% ({tr_loss:.4f}) | "
              f"val {va_acc*100:.2f}% ({va_loss:.4f}) | {elapsed:.0f}s", flush=True)

        if va_acc > best_val:
            best_val   = va_acc
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> phase best val {va_acc*100:.2f}%", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop (no val improvement for {PATIENCE} epochs)", flush=True)
                break

    # Restore best state from this phase
    if best_state:
        model.load_state_dict(best_state)
    return best_val

# ── Run all phases ────────────────────────────────────────────────────────────
phases = [
    ("Phase A — Head warmup (backbone frozen)",       EPOCHS_WARMUP, 0, HEAD_LR, 0.0),
    ("Phase B1 — Unfreeze last 1 backbone block",     EPOCHS_STAGE,  1, HEAD_LR, BACKBONE_LR),
    ("Phase B2 — Unfreeze last 2 backbone blocks",    EPOCHS_STAGE,  2, HEAD_LR, BACKBONE_LR),
    ("Phase B3 — Unfreeze last 3 backbone blocks",    EPOCHS_STAGE,  3, HEAD_LR, BACKBONE_LR),
    ("Phase B4 — Unfreeze last 4 backbone blocks",    EPOCHS_STAGE,  4, HEAD_LR, BACKBONE_LR * 0.5),
    ("Phase B5 — Unfreeze full backbone",             EPOCHS_STAGE,  9, HEAD_LR, BACKBONE_LR * 0.1),
]

for phase_args in phases:
    train_phase(*phase_args)

# ── Final evaluation ──────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("  Final Evaluation on Test Set", flush=True)
print(f"{'='*60}", flush=True)

model.eval()
all_preds, all_labels, top3_ok = [], [], 0
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        out = model(imgs.to(DEVICE))
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
        for pred3, lbl in zip(out.topk(3, dim=1).indices.cpu(), labels):
            if lbl.item() in pred3.tolist():
                top3_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  test batch {i+1}/{len(test_loader)}", flush=True)

test_acc  = accuracy(all_preds, all_labels) * 100
top3_acc  = top3_ok / len(all_labels) * 100
print(f"\nTest Accuracy  : {test_acc:.2f}%", flush=True)
print(f"Top-3 Accuracy : {top3_acc:.2f}%", flush=True)

# ── Save only if better than current best ────────────────────────────────────
save_dict = {"model_state": model.state_dict(),
             "class_to_idx": class_to_idx,
             "idx_to_class": idx_to_class,
             "num_classes":  num_classes}

torch.save(save_dict, MODEL_PATH)
print(f"Saved: {MODEL_PATH}", flush=True)

if test_acc > overall_best_acc:
    torch.save(save_dict, BEST_PATH)
    print(f"NEW BEST: {overall_best_acc:.2f}% -> {test_acc:.2f}%  "
          f"(+{test_acc - overall_best_acc:.2f}%)  saved to best_model.pt", flush=True)
else:
    print(f"Did not beat current best ({overall_best_acc:.2f}%). "
          f"best_model.pt unchanged.", flush=True)

# ── Reports ───────────────────────────────────────────────────────────────────
target_names = [idx_to_class[i] for i in range(num_classes)]
report = classification_report(all_labels, all_preds,
                                target_names=target_names, zero_division=0)
print(report, flush=True)
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "classification_report.txt").write_text(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Deep Training Confusion Matrix  ({test_acc:.2f}%)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=100)
plt.close()

# ── Training history ──────────────────────────────────────────────────────────
hist = json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else []
prev = hist[-1]["test_acc"] if hist else None
run_num = (hist[-1]["run"] + 1) if hist else 1
hist.append({
    "run":      run_num,
    "date":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "script":   "deep_train.py",
    "test_acc": round(test_acc, 2),
    "top3_acc": round(top3_acc, 2),
    "epochs":   sum(p[1] for p in phases),
    "notes":    "Full backbone fine-tune — staged unfreeze, differential LR, BN frozen",
})
HIST_PATH.write_text(json.dumps(hist, indent=2))
if prev:
    print(f"\nAccuracy change: {prev:.2f}% -> {test_acc:.2f}% "
          f"({test_acc - prev:+.2f}%)", flush=True)
print(f"History: {len(hist)} total runs", flush=True)
gc.collect()
