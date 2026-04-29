"""
Partial Face Recognition - Accuracy-Boosted Training.

Phase 0  Feature extraction (cached, instant on re-run).
Phase 1  Head training on cached features + feature noise augmentation.
         Skipped automatically if best_model.pt already exists.
Phase 2  Fine-tune last 2 backbone blocks on original-only subset
         (~2K images per epoch = ~4-5 min/epoch on CPU, 5 epochs max).
Phase 3  Test-time evaluation with 3-view TTA (flip + brightness).
"""

import sys, gc, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_utils import (IMG_SIZE, FaceDataset, build_model,
                         load_checkpoint, eval_tf, train_tf, accuracy,
                         MEAN, STD)

DATASET_DIR  = ROOT / "partial_face_dataset"
CACHE_DIR    = ROOT / "features_cache"
MODEL_PATH   = ROOT / "partial_face_model.pt"
BEST_PATH    = ROOT / "best_model.pt"
RESULTS_DIR  = ROOT / "results"

BATCH_EXTRACT = 64
BATCH_HEAD    = 512
BATCH_FINE    = 8        # small batch for fine-tune (memory)
EPOCHS_HEAD   = 80
EPOCHS_FINE   = 5
LR_HEAD       = 3e-3
LR_FINE       = 5e-6
PATIENCE_HEAD = 12
PATIENCE_FINE = 4
FEAT_NOISE    = 0.02     # Gaussian noise std added to features during head training
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
meta = pd.read_csv(DATASET_DIR / "metadata.csv")

if "split" in meta.columns:
    train_df = meta[meta["split"] == "train"].reset_index(drop=True)
    val_df   = meta[meta["split"] == "val"].reset_index(drop=True)
    test_df  = meta[meta["split"] == "test"].reset_index(drop=True)
else:
    train_df, tmp   = train_test_split(meta, test_size=0.30,
                                       stratify=meta["identity"], random_state=SEED)
    val_df, test_df = train_test_split(tmp, test_size=0.50,
                                       stratify=tmp["identity"], random_state=SEED)

# Original-only subset for Phase 2 fine-tuning (10x fewer images, 10x faster/epoch)
orig_train_df = train_df[train_df.get("transformation", pd.Series(["original"]*len(train_df))) == "original"].reset_index(drop=True)
if orig_train_df.empty:
    orig_train_df = train_df

print(f"Train {len(train_df):,}  Val {len(val_df):,}  Test {len(test_df):,}")
print(f"Orig-only train: {len(orig_train_df):,} (used for Phase 2)")

classes      = sorted(meta["identity"].unique().tolist())
num_classes  = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

y_train       = train_df["identity"].map(class_to_idx).values
weights_arr   = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
print(f"Classes: {num_classes}")

# ── Phase 0: Feature extraction (cached) ─────────────────────────────────────
CACHE_DIR.mkdir(exist_ok=True)

def extract_features(df_split, split_name, backbone):
    feat_path  = CACHE_DIR / f"{split_name}_feats.pt"
    label_path = CACHE_DIR / f"{split_name}_labels.pt"
    if feat_path.exists() and label_path.exists():
        print(f"  [cache] {split_name}")
        return (torch.load(feat_path,  weights_only=True),
                torch.load(label_path, weights_only=True))
    loader = DataLoader(FaceDataset(df_split, DATASET_DIR, eval_tf, class_to_idx),
                        batch_size=BATCH_EXTRACT, shuffle=False, num_workers=0)
    feats_list, labels_list = [], []
    backbone.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            f = backbone.avgpool(backbone.features(imgs.to(DEVICE))).flatten(1)
            feats_list.append(f.cpu())
            labels_list.append(labels)
            done = min((i+1)*BATCH_EXTRACT, len(df_split))
            print(f"  extracting {split_name}: {done}/{len(df_split)}", end="\r")
    feats  = torch.cat(feats_list)
    labels = torch.cat(labels_list)
    torch.save(feats,  feat_path)
    torch.save(labels, label_path)
    print(f"  [saved] {split_name}: {feats.shape}                     ")
    return feats, labels

print("\n[Phase 0] Feature extraction...")
backbone = build_model(num_classes, pretrained=True).to(DEVICE)

train_feats, train_labels = extract_features(train_df, "train", backbone)
val_feats,   val_labels   = extract_features(val_df,   "val",   backbone)
test_feats,  test_labels  = extract_features(test_df,  "test",  backbone)

model = backbone
head  = model.classifier
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

def tensor_loader(feats, labels, batch_size, shuffle):
    ds = TensorDataset(feats.to(DEVICE), labels.to(DEVICE))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_feat_ldr = tensor_loader(train_feats, train_labels, BATCH_HEAD, True)
val_feat_ldr   = tensor_loader(val_feats,   val_labels,   BATCH_HEAD, False)

# ── Phase 1: Head training with feature noise ─────────────────────────────────
def run_head_epoch(loader, optimizer=None):
    head.train() if optimizer else head.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for feats, labels in loader:
            if optimizer and FEAT_NOISE > 0:
                feats = feats + torch.randn_like(feats) * FEAT_NOISE
            if optimizer:
                optimizer.zero_grad()
            out  = head(feats)
            loss = criterion(out, labels)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(feats)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(feats)
    return total_loss / total, correct / total

def training_loop(epochs, optimizer, scheduler, label, patience, run_fn,
                  train_ldr, val_ldr, skip_if_exists=False):
    if skip_if_exists and BEST_PATH.exists():
        print(f"  [skip] {label} — best_model.pt already exists")
        return {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}, 0.0

    history    = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    no_improve = 0
    best_val   = 0.0
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_fn(train_ldr, optimizer)
        va_loss, va_acc = run_fn(val_ldr)
        if hasattr(scheduler, 'step'):
            scheduler.step(va_loss) if isinstance(scheduler,
                optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train {tr_acc*100:.1f}% ({tr_loss:.4f}) | "
              f"val {va_acc*100:.1f}% ({va_loss:.4f}) | "
              f"{time.time()-t0:.1f}s")
        if va_acc > best_val:
            best_val   = va_acc
            no_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "idx_to_class": idx_to_class,
                        "num_classes":  num_classes}, BEST_PATH)
            print(f"  -> best saved ({va_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
    return history, best_val

for param in model.features.parameters():
    param.requires_grad = False

opt1   = optim.Adam(head.parameters(), lr=LR_HEAD, weight_decay=1e-4)
sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_HEAD, eta_min=1e-5)

history1, best1 = training_loop(
    EPOCHS_HEAD, opt1, sched1,
    f"Phase 1 - Head + feature noise  (lr={LR_HEAD})",
    PATIENCE_HEAD, run_head_epoch, train_feat_ldr, val_feat_ldr,
    skip_if_exists=True,   # skip if already trained
)

# ── Phase 2: Fine-tune last 2 backbone blocks on original images ───────────────
print(f"\n[Phase 2] Fine-tuning backbone on {len(orig_train_df):,} original images...")

ckpt = load_checkpoint(BEST_PATH, DEVICE)
model.load_state_dict(ckpt["model_state"])

# Unfreeze last 2 EfficientNet blocks + head
for param in model.parameters():
    param.requires_grad = False
# model.features has 9 children (0-8); unfreeze 7 and 8
for layer in list(model.features.children())[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable_params:,}")

orig_train_ldr = DataLoader(
    FaceDataset(orig_train_df, DATASET_DIR, train_tf, class_to_idx),
    batch_size=BATCH_FINE, shuffle=True, num_workers=0)
val_img_ldr = DataLoader(
    FaceDataset(val_df, DATASET_DIR, eval_tf, class_to_idx),
    batch_size=BATCH_FINE, shuffle=False, num_workers=0)

criterion_fine = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

def run_full_epoch(loader, optimizer=None):
    model.train() if optimizer else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if optimizer:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion_fine(out, labels)
            if optimizer:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(imgs)
    return total_loss / total, correct / total

opt2   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=LR_FINE, weight_decay=1e-4)
sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_FINE, eta_min=1e-7)

history2, _ = training_loop(
    EPOCHS_FINE, opt2, sched2,
    f"Phase 2 - Backbone fine-tune  (lr={LR_FINE})",
    PATIENCE_FINE, run_full_epoch, orig_train_ldr, val_img_ldr,
    skip_if_exists=False,  # always run Phase 2
)

# ── Phase 3: Evaluate with 3-view TTA ────────────────────────────────────────
print("\n[Phase 3] Evaluating with 3-view TTA...")
ckpt = load_checkpoint(BEST_PATH, DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Build TTA transforms: original, h-flip, brightness boost
tta_tfs = [
    eval_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]),
]

from PIL import Image as PILImage
import cv2 as _cv2

def tta_predict_batch(img_paths):
    all_probs = []
    for tf in tta_tfs:
        probs_list = []
        for path in img_paths:
            img = _cv2.imread(str(path))
            if img is None:
                probs_list.append(np.zeros(num_classes))
                continue
            pil = PILImage.fromarray(_cv2.cvtColor(img, _cv2.COLOR_BGR2RGB))
            tensor = tf(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()
            probs_list.append(p)
        all_probs.append(np.stack(probs_list))  # (N, num_classes)
    # Average across TTA views
    return np.mean(all_probs, axis=0)  # (N, num_classes)

# Evaluate on test set with TTA
all_preds, all_labels_list, top3_correct = [], test_labels.tolist(), 0
EVAL_BATCH = 32

for start in range(0, len(test_df), EVAL_BATCH):
    batch = test_df.iloc[start:start+EVAL_BATCH]
    paths = [DATASET_DIR / row["filename"] for _, row in batch.iterrows()]
    probs = tta_predict_batch(paths)
    preds = probs.argmax(1).tolist()
    all_preds.extend(preds)
    for p_arr, true_lbl in zip(probs, test_labels[start:start+EVAL_BATCH].tolist()):
        if true_lbl in p_arr.argsort()[-3:].tolist():
            top3_correct += 1
    done = min(start + EVAL_BATCH, len(test_df))
    print(f"  TTA eval: {done}/{len(test_df)}", end="\r")

print(f"\nTest Accuracy  (TTA): {accuracy(all_preds, all_labels_list)*100:.2f}%")
print(f"Top-3 Accuracy (TTA): {top3_correct/len(all_labels_list)*100:.2f}%")

target_names = [idx_to_class[i] for i in range(num_classes)]
report = classification_report(all_labels_list, all_preds,
                               target_names=target_names, zero_division=0)
print(report)
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "classification_report.txt").write_text(report)

cm = confusion_matrix(all_labels_list, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=100); plt.close()

if "transformation" in test_df.columns:
    results = test_df.assign(pred=all_preds, true=all_labels_list)
    results["correct"] = results["pred"] == results["true"]
    print("\n--- Accuracy by Occlusion Type ---")
    for t in sorted(results["transformation"].unique()):
        sub = results[results["transformation"] == t]
        print(f"  {t:<22} {sub['correct'].mean()*100:.1f}%  ({len(sub)} samples)")

# Combined learning curve
all_h  = {k: history1[k] + history2[k] for k in history1}
offset = len(history1["train_acc"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, title in [(axes[0], "acc", "Accuracy"), (axes[1], "loss", "Loss")]:
    if all_h[f"train_{key}"]:
        ax.plot(all_h[f"train_{key}"], label="Train")
        ax.plot(all_h[f"val_{key}"],   label="Val")
        if offset > 1:
            ax.axvline(offset - 1, color="gray", linestyle="--", label="Fine-tune start")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
plt.tight_layout(); plt.savefig(RESULTS_DIR / "learning_curve_accuracy.png", dpi=100); plt.close()

torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class, "num_classes": num_classes}, MODEL_PATH)
print(f"\nModel : {MODEL_PATH}\nBest  : {BEST_PATH}")

import json as _json, datetime as _dt
_hist_path = RESULTS_DIR / "training_history.json"
_hist = _json.loads(_hist_path.read_text()) if _hist_path.exists() else []
_hist.append({
    "run":      len(_hist) + 1,
    "date":     _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "script":   "pretrained_model.py",
    "test_acc": round(accuracy(all_preds, all_labels_list) * 100, 2),
    "top3_acc": round(top3_correct / len(all_labels_list) * 100, 2),
    "epochs":   len(history1["train_acc"]) + len(history2["train_acc"]),
    "notes":    f"Phase 1 ({len(train_df):,} cached) + Phase 2 ({len(orig_train_df):,} originals)",
})
_hist_path.write_text(_json.dumps(_hist, indent=2))
print(f"History saved ({len(_hist)} runs total)")
gc.collect()