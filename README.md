# Partial Face Recognition System

A closed-set face recognition system that correctly identifies people even when their face is **partially occluded** — masked, blurred, cropped, or obscured. Built with **PyTorch + EfficientNet-B0** transfer learning and a **Streamlit** web interface.

Developed as a university deep learning course project.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy (Top-1) | ~88–89% |
| Top-3 Accuracy | ~94–95% |
| Classes | 100 CelebA celebrities |
| Training images | 21,266 (100 identities × 10 occlusion variants) |

**Accuracy by occlusion type:**

| Occlusion | Accuracy |
|-----------|----------|
| Original (clean) | ~99% |
| Sunglasses | ~98% |
| Noise patch | ~96% |
| Random block | ~97% |
| Crop (left/right) | ~91% |
| Surgical mask | ~87% |
| Top crop | ~78% |
| Blurred | ~43% |

---

## Project Structure

```
partial-face-recognition-main/
├── App/
│   └── steamlit_app.py          # Streamlit web application
├── Model/
│   ├── pretrained_model.py      # Phase 1 — build feature cache + train head
│   ├── Main.py                  # Phase 2 — head fine-tuning
│   ├── predict.py               # CLI prediction script
│   ├── deep_train.py            # Full backbone fine-tuning (staged unfreeze)
│   └── night_train.py           # Overnight head-only training loop
├── assets/
│   └── deploy.prototxt          # OpenCV DNN face detector config
├── results/
│   ├── confusion_matrix.png     # Confusion matrix heatmap
│   ├── learning_curve_accuracy.png  # Training accuracy curve
│   ├── classification_report.txt    # Per-identity precision / recall / F1
│   └── training_history.json        # Log of all training runs
├── model_utils.py               # Shared utilities (model arch, dataset, transforms)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Dataset Sample

A minimal sample is included in [`dataset_sample/`](./dataset_sample/) so you can explore the data structure without downloading the full dataset.

```
dataset_sample/
├── Female_Adult_Blond_08/
│   ├── Female_Adult_Blond_08_000_original.jpg
│   ├── Female_Adult_Blond_08_000_blurred.jpg
│   ├── Female_Adult_Blond_08_000_sunglasses.jpg
│   └── ... (10 occlusion variants)
├── Female_Adult_Dark_Hair_09/
│   └── ... (10 occlusion variants)
├── Male_Adult_Black_Hair_93/
│   └── ... (10 occlusion variants)
└── metadata.csv
```

**3 identities · 10 occlusion types each · 30 images total**

Each image in the full dataset exists in 10 variants:

| Occlusion | Description |
|-----------|-------------|
| `original` | Clean, unmodified face |
| `blurred` | Gaussian blur applied |
| `sunglasses` | Eyes blocked |
| `surgical_mask` | Lower face covered |
| `top_crop` | Top half removed |
| `bottom_crop` | Bottom half removed |
| `left_crop` | Left half removed |
| `right_crop` | Right half removed |
| `noise_patch` | Centre replaced with noise |
| `random_block` | Black rectangle over face |

The full dataset (100 identities × ~21 images × 10 variants = ~21,000 images) is not included due to size (502 MB) and CelebA licensing.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download required assets

**Dataset** — place the `partial_face_dataset/` folder in the project root.  
It must contain one subfolder per identity (e.g. `Female_Adult_Blond_08/`) with images and a `metadata.csv` at the root with columns `identity`, `filename`, `transformation`, `split`.

**Face detector** — download the OpenCV ResNet-SSD weights (11 MB):
```
res10_300x300_ssd_iter_140000.caffemodel
```
Place it alongside `deploy.prototxt` in the project root. These files enable accurate face detection in uploaded photos.

### 3. Train the model

**Phase 1 — build feature cache and train head** (~20–30 min, run once):
```bash
python Model/pretrained_model.py
```
This extracts EfficientNet-B0 features for all 21k images, caches them to `features_cache/`, trains the classifier head, and saves `best_model.pt`.

**Phase 2 — fine-tune the head** (optional, ~5–15 min):
```bash
python Model/Main.py
```

**Overnight head loop** (optional, runs until 08:00):
```bash
python night_train.py
```

**Full backbone fine-tuning** (optional, 4–8 hrs on CPU):
```bash
python deep_train.py
```

### 4. Launch the web app

```bash
streamlit run App/steamlit_app.py
```

Open [localhost:8501](http://localhost:8501) in your browser.

---

## Web App Pages

| Page | Description |
|------|-------------|
| **Recognize** | Upload a face image and get top-K predictions with confidence scores |
| **Dataset** | Browse all 100 identities and their occlusion variants |
| **Add Data** | Upload new face photos and assign them to an identity |
| **Evaluations** | Confusion matrix, per-identity F1, occlusion breakdown, radar chart |
| **Train & History** | Launch training scripts and view accuracy history across all runs |

---

## Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained, frozen during head training)
- **Head:** `Dropout(0.4) → Linear(1280→512) → BN → ReLU → Dropout(0.3) → Linear(512→100)`
- **Loss:** CrossEntropyLoss with label smoothing (0.05–0.08) + class weighting
- **Face detection:** OpenCV ResNet-SSD (primary) with Haar cascade fallback
- **Inference:** Temperature scaling (T=0.5) applied to logits for sharper confidence

---

## Notes

- The model is a **closed-set classifier** — it can only identify the 100 people it was trained on. Photos of anyone else will produce a low-confidence result (shown with a warning at <40% confidence).
- `features_cache/` and `partial_face_dataset/` are excluded from the repository (too large). Run Phase 1 training to regenerate the cache.
- `best_model.pt` is also excluded. Run training to produce it, or ask for a pre-trained weights file.

---

## Credits

Developed by:

- **Raja Ali Tabish**
- **Aatma Ram**
- **Moiz Kiani**
- **Aparna Ghimire**

Supervised as part of a university deep learning course.

---

## License

MIT License — see [LICENSE](./LICENSE).
