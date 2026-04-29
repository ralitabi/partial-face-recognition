"""
Continue fine-tuning best_model.pt on the original (non-augmented) face subset.

Why original-only?
  Full training set has 21,266 images x 10 occlusion variants.
  Running all of them through EfficientNet with gradients takes ~20-30 min/epoch.
  The 'original' subset has only ~2,100 images  -> ~4-5 min/epoch on CPU.
  Fine-tuning on clean faces adapts the backbone correctly; occlusion
  robustness is already baked in from Phase 1 head training.

Usage:
    python Model/Main.py
"""

import sys, gc, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_utils import (FaceDataset, build_model, load_checkpoint,
                         train_tf, eval_tf, accuracy)

DATASET_DIR = ROOT / "partial_face_dataset"
MODEL_PATH  = ROOT / "partial_face_model.pt"
BEST_PATH   = ROOT / "best_model.pt"
RESULTS_DIR = ROOT / "results"
HIST_PATH   = RESULTS_DIR / "training_history.json"

BATCH_SIZE = 8      # small — full backbone + gradients on CPU needs low memory
EPOCHS     = 10
LR         = 5e-6   # very small — backbone is already well-trained
PATIENCE   = 4
SEED       = 42

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

path = BEST_PATH if BEST_PATH.exists() else MODEL_PATH
if not path.exists():
    print("No model found. Run pretrained_model.py first.")
    sys.exit(1)

ckpt         = load_checkpoint(path, DEVICE)
num_classes  = ckpt["num_classes"]
class_to_idx = ckpt["class_to_idx"]
idx_to_class = ckpt["idx_to_class"]
print(f"Loaded: {path.name}  ({num_classes} classes)", flush=True)

# ── Use original images only ───────────────────────────────────────────────────
meta     = pd.read_csv(DATASET_DIR / "metadata.csv")
train_df = meta[meta["split"] == "train"].reset_index(drop=True)
val_df   = meta[meta["split"] == "val"].reset_index(drop=True)
test_df  = meta[meta["split"] == "test"].reset_index(drop=True)

if "transformation" in meta.columns:
    orig_train = train_df[train_df["transformation"] == "original"].reset_index(drop=True)
else:
    orig_train = train_df

print(f"Using {len(orig_train):,} original train images  "
      f"(full set has {len(train_df):,})", flush=True)
print(f"Val {len(val_df):,}  Test {len(test_df):,}", flush=True)

y_train       = orig_train["identity"].map(class_to_idx).values
weights_arr   = compute_class_weight("balanced",
                                     classes=np.arange(num_classes), y=y_train)
class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)

train_loader = DataLoader(FaceDataset(orig_train, DATASET_DIR, train_tf, class_to_idx),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(FaceDataset(val_df,     DATASET_DIR, eval_tf,  class_to_idx),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(FaceDataset(test_df,    DATASET_DIR, eval_tf,  class_to_idx),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model — head only (backbone frozen to prevent catastrophic forgetting) ─────
# Evidence: unlocking backbone on 2k images collapsed accuracy from 86% → 17%.
model = build_model(num_classes, pretrained=True).to(DEVICE)
# Load only the classifier weights from checkpoint; keep pretrained backbone.
head_weights = {k: v for k, v in ckpt["model_state"].items()
                if k.startswith("classifier.")}
model.load_state_dict(head_weights, strict=False)

for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {n_trainable:,} / {n_total:,} "
      f"({n_trainable/n_total*100:.1f}%  — head only)", flush=True)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
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
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(imgs)
            # Print progress every 20 batches so user knows it's running
            if train and (batch_idx + 1) % 20 == 0:
                print(f"    batch {batch_idx+1}/{len(loader)}  "
                      f"loss={total_loss/total:.4f}  "
                      f"acc={correct/total*100:.1f}%", flush=True)
    return total_loss / total, correct / total


print(f"\n{'='*58}")
print(f"  Fine-tune last 3 blocks  "
      f"(lr={LR}, max {EPOCHS} epochs, batch={BATCH_SIZE})")
print(f"{'='*58}", flush=True)

# Initialise from current best so we never overwrite a better model
import json as _json
_hist      = _json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else []
_hist_best = max((_h["test_acc"] for _h in _hist), default=0.0) / 100.0
# Use a conservative floor — only save if we beat the historical best val
# (approximated by using 85 % of test_acc as a val_acc floor)
best_val_acc = max(0.0, _hist_best * 0.85)
print(f"Starting best_val_acc floor: {best_val_acc*100:.2f}%  "
      f"(historical best test acc: {_hist_best*100:.2f}%)", flush=True)

no_improve = 0
history    = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS} — training...", flush=True)
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    print(f"Epoch {epoch}/{EPOCHS} — validating...", flush=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    elapsed = time.time() - t0
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"train {tr_acc*100:.1f}% ({tr_loss:.4f}) | "
          f"val {va_acc*100:.1f}% ({va_loss:.4f}) | "
          f"{elapsed:.0f}s", flush=True)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        no_improve   = 0
        torch.save({"model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "num_classes":  num_classes}, BEST_PATH)
        print(f"  -> best saved ({va_acc*100:.2f}%)", flush=True)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...", flush=True)
model.eval()
all_preds, all_labels, top3_correct = [], [], 0
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        out = model(imgs.to(DEVICE))
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())
        for pred3, lbl in zip(out.topk(3, dim=1).indices.cpu(), labels):
            if lbl.item() in pred3.tolist():
                top3_correct += 1
        if (i + 1) % 50 == 0:
            print(f"  test batch {i+1}/{len(test_loader)}", flush=True)

print(f"\nTest Accuracy  : {accuracy(all_preds, all_labels)*100:.2f}%")
print(f"Top-3 Accuracy : {top3_correct/len(all_labels)*100:.2f}%")

target_names = [idx_to_class[i] for i in range(num_classes)]
report = classification_report(all_labels, all_preds,
                               target_names=target_names, zero_division=0)
print(report)
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "classification_report.txt").write_text(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Fine-Tuned Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=100)
plt.close()

if "transformation" in test_df.columns:
    results = test_df.assign(pred=all_preds, true=all_labels)
    results["correct"] = results["pred"] == results["true"]
    print("\n--- Accuracy by Occlusion Type ---")
    for t in sorted(results["transformation"].unique()):
        sub = results[results["transformation"] == t]
        print(f"  {t:<22} {sub['correct'].mean()*100:.1f}%  ({len(sub)} samples)")

plt.figure(figsize=(10, 4))
plt.plot(history["train_acc"], label="Train")
plt.plot(history["val_acc"],   label="Val")
plt.title("Fine-Tuning Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fine_tune_accuracy.png", dpi=100)
plt.close()

torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class, "num_classes": num_classes}, MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}", flush=True)

import json as _json, datetime as _dt
_hist     = _json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else []
_prev_acc = _hist[-1]["test_acc"] if _hist else None
_cur_acc  = accuracy(all_preds, all_labels) * 100
_hist.append({
    "run":      len(_hist) + 1,
    "date":     _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "script":   "Main.py",
    "test_acc": round(_cur_acc, 2),
    "top3_acc": round(top3_correct / len(all_labels) * 100, 2),
    "epochs":   len(history["train_acc"]),
    "notes":    f"Head fine-tune on {len(orig_train):,} originals",
})
HIST_PATH.write_text(_json.dumps(_hist, indent=2))
if _prev_acc:
    print(f"Accuracy change: {_prev_acc:.2f}% -> {_cur_acc:.2f}% "
          f"({_cur_acc - _prev_acc:+.2f}%)", flush=True)
print(f"History saved ({len(_hist)} runs total)", flush=True)
gc.collect()
