# Partial Face Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-3fb950?style=flat-square)

A deep learning–based closed-set face recognition system designed to identify individuals even when faces are **partially occluded** — masks, blur, cropping, or noise.

Built with **PyTorch + EfficientNet-B0** transfer learning and a **Streamlit** web interface for interactive testing and evaluation.

> Developed as part of a university deep learning project.

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training Pipeline](#training-pipeline)
- [Web Application](#web-application)
- [Model Architecture](#model-architecture)
- [Notes](#notes)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This project focuses on improving face recognition performance under challenging real-world conditions where facial features may be partially hidden. The system uses **transfer learning** and a structured multi-phase training pipeline to maintain robustness across ten distinct occlusion types.

---

## Results

### Overall Performance

| Metric | Score |
|---|---|
| Top-1 Accuracy | **~88–89%** |
| Top-3 Accuracy | **~94–95%** |
| Number of Classes | 100 |
| Training Images | ~21,266 |

### Accuracy by Occlusion Type

| Occlusion | Accuracy |
|---|---|
| Original (clean) | ~99% |
| Sunglasses | ~98% |
| Noise Patch | ~96% |
| Random Block | ~97% |
| Crop (Left / Right) | ~91% |
| Surgical Mask | ~87% |
| Top Crop | ~78% |
| Blurred | ~43% |

---

## Project Structure

```
partial-face-recognition/
├── App/
│   └── steamlit_app.py           # Streamlit web interface
├── Model/
│   ├── pretrained_model.py       # Phase 1 — feature extraction + head training
│   ├── Main.py                   # Phase 2 — head fine-tuning
│   ├── deep_train.py             # Full backbone training (staged unfreeze)
│   ├── night_train.py            # Overnight head training loop
│   └── predict.py                # CLI inference script
├── assets/
│   ├── deploy.prototxt           # OpenCV DNN face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── results/
│   ├── confusion_matrix.png
│   ├── learning_curve_accuracy.png
│   ├── classification_report.txt
│   └── training_history.json
├── dataset_sample/               # 3 identities x 10 occlusion types (30 images)
├── model_utils.py                # Shared model utilities
├── requirements.txt
└── README.md
```

---

## Dataset

A small sample dataset is included under [`dataset_sample/`](./dataset_sample/) to demonstrate the expected data format without requiring the full download.

```
dataset_sample/
├── Female_Adult_Blond_08/        # 10 occlusion variants
├── Female_Adult_Dark_Hair_09/    # 10 occlusion variants
├── Male_Adult_Black_Hair_93/     # 10 occlusion variants
└── metadata.csv
```

### Occlusion Types

| Type | Description |
|---|---|
| `original` | Clean, unmodified face |
| `blurred` | Gaussian blur applied |
| `sunglasses` | Eyes covered by dark rectangle |
| `surgical_mask` | Lower face covered |
| `top_crop` | Upper half of face removed |
| `bottom_crop` | Lower half of face removed |
| `left_crop` | Left side removed |
| `right_crop` | Right side removed |
| `noise_patch` | Centre replaced with random noise |
| `random_block` | Solid black rectangle over face |

> The full dataset (~500 MB, 100 identities) is excluded due to size and CelebA licensing.
> The `metadata.csv` must contain columns: `identity`, `filename`, `transformation`, `split`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download required files

Place the following in the `assets/` folder:

```
assets/res10_300x300_ssd_iter_140000.caffemodel
```

This is the OpenCV ResNet-SSD face detector (~11 MB, available from the OpenCV repository).

### 3. Add the full dataset

Place the full dataset at the project root:

```
partial_face_dataset/
└── <identity_name>/
    ├── <identity_name>_000_original.jpg
    ├── <identity_name>_000_blurred.jpg
    └── ...
```

---

## Training Pipeline

Training is split into phases. Run them in order for best results.

### Phase 1 — Feature Extraction + Head Training

Extracts EfficientNet-B0 features for all images, caches them, then trains the classifier head. Run once (~20–30 min on CPU).

```bash
python Model/pretrained_model.py
```

### Phase 2 — Head Fine-Tuning

Continues training the head with a lower learning rate. Safe and fast (~5–15 min).

```bash
python Model/Main.py
```

### Full Backbone Training *(optional)*

Staged backbone unfreeze with differential learning rates. Targets 90%+ accuracy. Slow (~4–8 hrs on CPU).

```bash
python Model/deep_train.py
```

### Overnight Training Loop *(optional)*

Cycles through 13 hyperparameter configurations until 08:00. Best for overnight runs.

```bash
python Model/night_train.py
```

---

## Web Application

Launch the Streamlit interface:

```bash
streamlit run App/steamlit_app.py
```

Open in your browser at `http://localhost:8501`

### Pages

| Page | Description |
|---|---|
| **Recognize** | Upload a face image — get top-K predictions with confidence scores |
| **Dataset** | Browse all 100 identities and their occlusion variants |
| **Add Data** | Upload new face photos and assign them to an identity |
| **Evaluations** | Confusion matrix, per-identity F1, occlusion breakdown, radar chart |
| **Train & History** | Launch training scripts and view accuracy across all runs |

---

## Model Architecture

```
Input Image (224x224)
       |
EfficientNet-B0 Backbone  <-- ImageNet pretrained, frozen during head training
       |
  [1280-dim features]
       |
  Dropout (p=0.4)
  Linear (1280 -> 512)
  BatchNorm1D
  ReLU
  Dropout (p=0.3)
  Linear (512 -> 100)
       |
  Class Logits
       |
Temperature Scaling (T=0.5)  <-- sharpens confidence scores at inference
       |
    Softmax
```

- **Loss:** CrossEntropyLoss with label smoothing (0.05–0.08) + class weighting
- **Face Detection:** OpenCV ResNet-SSD (primary) → Haar cascade (fallback)
- **Optimizer:** Adam with cosine annealing

---

## Notes

- This is a **closed-set** recognition system — it can only identify the 100 people it was trained on
- Uploading a photo of someone outside those 100 will produce a **low-confidence** result (flagged at < 40%)
- Model weights (`.pt`) and the full dataset are excluded from the repository — run training to generate them
- The feature cache (`features_cache/`) is regenerated automatically by Phase 1 training

---

## Future Work

- Open-set recognition with rejection thresholds
- Improved accuracy on heavily blurred images
- Real-time video inference
- Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
- Expanded dataset with greater demographic diversity

---

## Contributors

**Raja Ali Tabish**

---

## License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.
