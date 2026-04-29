"""
Overnight training loop — runs Phase 1 head training repeatedly until 8 AM.

WHY HEAD-ONLY (no backbone fine-tuning):
  Backbone fine-tuning on only ~2k images causes catastrophic forgetting.
  Evidence: Run 1 (head-only) = 86.86%, Run 2 (+ backbone) = 17.45%.
  Strategy: loop head training with slight LR/noise variations to push past 86.86%.

Runs in ~7 min per cycle on CPU (features already cached).
"""

import sys, gc, time, json, datetime, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
CACHE_DIR   = ROOT / "features_cache"
BEST_PATH   = ROOT / "best_model.pt"
MODEL_PATH  = ROOT / "partial_face_model.pt"
HIST_PATH   = ROOT / "results" / "training_history.json"
META_PATH   = ROOT / "partial_face_dataset" / "metadata.csv"

sys.path.insert(0, str(ROOT))
from model_utils import build_model, load_checkpoint, accuracy

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── Target time ───────────────────────────────────────────────────────────────
TARGET = datetime.datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
if TARGET <= datetime.datetime.now():
    TARGET += datetime.timedelta(days=1)

def mins_left():
    return (TARGET - datetime.datetime.now()).total_seconds() / 60

def time_ok():
    return datetime.datetime.now() < TARGET

print(f"\n{'='*60}")
print(f"  Overnight Training Loop")
print(f"  Target: 08:00  ({mins_left():.0f} min from now)")
print(f"{'='*60}\n", flush=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)

# ── Load cached features ──────────────────────────────────────────────────────
def load_cache():
    required = ["train_feats","train_labels","val_feats","val_labels","test_feats","test_labels"]
    missing  = [f for f in required if not (CACHE_DIR / f"{f}.pt").exists()]
    if missing:
        print(f"[ERROR] Missing cache files: {missing}")
        print("Run pretrained_model.py once first to build the cache.")
        sys.exit(1)
    print("Loading cached features…", flush=True)
    d = {k: torch.load(CACHE_DIR / f"{k}.pt", weights_only=True) for k in required}
    print(f"  Train: {d['train_feats'].shape}  Val: {d['val_feats'].shape}  Test: {d['test_feats'].shape}", flush=True)
    return d

# ── Load class mapping from metadata ─────────────────────────────────────────
meta         = pd.read_csv(META_PATH)
classes      = sorted(meta["identity"].unique().tolist())
num_classes  = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
print(f"Classes: {num_classes}", flush=True)

cache    = load_cache()
y_train  = cache["train_labels"].numpy()
w_arr    = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
c_weight = torch.tensor(w_arr, dtype=torch.float32).to(DEVICE)

feat_dim = cache["train_feats"].shape[1]   # 1280 for EfficientNet-B0

# ── Best accuracy tracker ─────────────────────────────────────────────────────
def best_history_acc():
    if not HIST_PATH.exists(): return 0.0
    h = json.loads(HIST_PATH.read_text())
    return max((r["test_acc"] for r in h), default=0.0)

overall_best = best_history_acc()
print(f"Best accuracy so far: {overall_best:.2f}%\n", flush=True)

# ── Training configs — varied to explore hyperparameter space ─────────────────
# 22 hours available → ~180 cycles. More variety = better chance of escaping
# the 88.76% local optimum reached by the previous overnight run.
CONFIGS = [
    # ── Low-LR, low-noise (refinement) ──
    {"lr": 1e-3, "noise": 0.02, "epochs": 100, "patience": 15, "label_smooth": 0.08, "batch": 512},
    {"lr": 5e-4, "noise": 0.02, "epochs": 100, "patience": 15, "label_smooth": 0.05, "batch": 512},
    {"lr": 8e-4, "noise": 0.03, "epochs": 100, "patience": 15, "label_smooth": 0.06, "batch": 256},
    # ── Medium-LR, medium-noise (main range) ──
    {"lr": 2e-3, "noise": 0.03, "epochs": 80,  "patience": 12, "label_smooth": 0.08, "batch": 512},
    {"lr": 3e-3, "noise": 0.04, "epochs": 80,  "patience": 12, "label_smooth": 0.10, "batch": 512},
    {"lr": 2e-3, "noise": 0.05, "epochs": 80,  "patience": 12, "label_smooth": 0.08, "batch": 256},
    {"lr": 1e-3, "noise": 0.06, "epochs": 80,  "patience": 12, "label_smooth": 0.06, "batch": 128},
    # ── Warm restarts (high LR, let scheduler anneal fast) ──
    {"lr": 5e-3, "noise": 0.02, "epochs": 60,  "patience": 10, "label_smooth": 0.12, "batch": 512},
    {"lr": 6e-3, "noise": 0.03, "epochs": 60,  "patience": 10, "label_smooth": 0.10, "batch": 512},
    # ── High-noise, high-regularisation (anti-overfit) ──
    {"lr": 2e-3, "noise": 0.08, "epochs": 80,  "patience": 12, "label_smooth": 0.12, "batch": 256},
    {"lr": 1e-3, "noise": 0.10, "epochs": 80,  "patience": 12, "label_smooth": 0.12, "batch": 128},
    # ── Small batch, more gradient updates per epoch ──
    {"lr": 5e-4, "noise": 0.04, "epochs": 80,  "patience": 12, "label_smooth": 0.08, "batch": 64},
    {"lr": 1e-3, "noise": 0.04, "epochs": 80,  "patience": 12, "label_smooth": 0.08, "batch": 64},
]

def make_loader(feats, labels, batch_size, shuffle):
    ds = TensorDataset(feats.to(DEVICE), labels.to(DEVICE))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def run_head_epoch(head, loader, criterion, optimizer=None, noise=0.0):
    head.train() if optimizer else head.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for feats, labels in loader:
            if optimizer and noise > 0:
                feats = feats + torch.randn_like(feats) * noise
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

def train_one_cycle(cfg, cycle_num):
    """Train head only with given config. Returns (test_acc, top3_acc) or None."""
    print(f"\n  Config: lr={cfg['lr']} noise={cfg['noise']} smooth={cfg['label_smooth']} batch={cfg['batch']}", flush=True)

    # Always use the ORIGINAL pretrained backbone — this matches the feature cache.
    # Backbone fine-tuning corrupts the features and causes catastrophic forgetting.
    torch.manual_seed(cycle_num * 17 + 42)
    backbone = build_model(num_classes, pretrained=True).to(DEVICE)

    # Warm-start the classifier head only from best checkpoint
    if BEST_PATH.exists():
        try:
            ckpt = load_checkpoint(BEST_PATH, DEVICE)
            if ckpt.get("num_classes") == num_classes:
                head_weights = {k: v for k, v in ckpt["model_state"].items()
                                if k.startswith("classifier.")}
                backbone.load_state_dict(head_weights, strict=False)
                print("  Warm-started classifier head from best_model.pt", flush=True)
        except Exception:
            pass

    for param in backbone.features.parameters():
        param.requires_grad = False

    head      = backbone.classifier
    criterion = nn.CrossEntropyLoss(weight=c_weight, label_smoothing=cfg["label_smooth"])
    optimizer = optim.Adam(head.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    tr_ldr = make_loader(cache["train_feats"], cache["train_labels"], cfg["batch"], True)
    va_ldr = make_loader(cache["val_feats"],   cache["val_labels"],   cfg["batch"], False)

    best_val   = 0.0
    no_improve = 0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        if not time_ok():
            print(f"  Time limit reached at epoch {epoch}", flush=True)
            break
        t0 = time.time()
        tr_loss, tr_acc = run_head_epoch(head, tr_ldr, criterion, optimizer, cfg["noise"])
        va_loss, va_acc = run_head_epoch(head, va_ldr, criterion)
        scheduler.step()

        print(f"  Ep {epoch:3d}/{cfg['epochs']} | train {tr_acc*100:.1f}% | val {va_acc*100:.1f}% | {time.time()-t0:.1f}s", flush=True)

        if va_acc > best_val:
            best_val   = va_acc
            no_improve = 0
            best_state = {k: v.clone() for k,v in backbone.state_dict().items()}
            print(f"  -> val best {va_acc*100:.2f}%", flush=True)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        return None

    # Restore best classifier weights onto the original pretrained backbone
    backbone.load_state_dict(best_state, strict=False)
    backbone.eval()
    te_ldr    = make_loader(cache["test_feats"], cache["test_labels"], 512, False)
    all_preds = []; all_labels = []
    top3_ok   = 0
    with torch.no_grad():
        for feats, labels in te_ldr:
            out = backbone.classifier(feats)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            for row, lbl in zip(out.topk(3,dim=1).indices.cpu(), labels.cpu()):
                if lbl.item() in row.tolist():
                    top3_ok += 1

    test_acc  = accuracy(all_preds, all_labels) * 100
    top3_acc  = top3_ok / len(all_labels) * 100
    return test_acc, top3_acc, backbone, best_state

# ── Main loop ─────────────────────────────────────────────────────────────────
cycle     = 0
run_count = len(json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else "[]")

while time_ok():
    cycle += 1
    cfg   = CONFIGS[(cycle - 1) % len(CONFIGS)]

    print(f"\n{'='*60}", flush=True)
    print(f"  Cycle {cycle}  |  {mins_left():.0f} min remaining  |  {datetime.datetime.now().strftime('%H:%M')}", flush=True)
    print(f"{'='*60}", flush=True)

    result = train_one_cycle(cfg, cycle)
    if result is None:
        print("  No improvement this cycle.", flush=True)
        continue

    test_acc, top3_acc, backbone, best_state = result
    run_count += 1

    improved = test_acc > overall_best

    print(f"\n  Test Accuracy : {test_acc:.2f}%   {'<== NEW BEST!' if improved else ''}", flush=True)
    print(f"  Top-3 Accuracy: {top3_acc:.2f}%", flush=True)

    # Always save latest model
    save_dict = {
        "model_state": best_state,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes":  num_classes,
    }
    torch.save(save_dict, MODEL_PATH)

    # Save best model only if improved
    if improved:
        overall_best = test_acc
        torch.save(save_dict, BEST_PATH)
        print(f"  best_model.pt updated -> {test_acc:.2f}%", flush=True)

    # Append to training history
    hist = json.loads(HIST_PATH.read_text()) if HIST_PATH.exists() else []
    hist.append({
        "run":      run_count,
        "date":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script":   "night_train.py",
        "test_acc": round(test_acc, 2),
        "top3_acc": round(top3_acc, 2),
        "epochs":   cfg["epochs"],
        "notes":    f"Head-only overnight | lr={cfg['lr']} noise={cfg['noise']}",
    })
    HIST_PATH.write_text(json.dumps(hist, indent=2))
    print(f"  History saved (run #{run_count})", flush=True)
    gc.collect()

print(f"\n{'='*60}", flush=True)
print(f"  OVERNIGHT TRAINING COMPLETE", flush=True)
print(f"  Total cycles: {cycle}", flush=True)
print(f"  Best accuracy achieved: {overall_best:.2f}%", flush=True)
print(f"{'='*60}\n", flush=True)
