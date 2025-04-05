# Partial Face Recognition System

This repository presents a comprehensive project on **Partial Face Recognition** using deep learning. The goal is to accurately recognize a person’s identity even when their face is partially occluded due to
masks, cropping, blurring, or noise. The solution is built using **TensorFlow/Keras**, utilizes **EfficientNetB0** for feature extraction, and offers a user-friendly web interface using **Streamlit**.
This project was designed and developed as part of a deep learning course, demonstrating strong teamwork, real-world problem-solving, and an end-to-end deployment pipeline.

## Table of Contents
- [Features](#Features)
- [Repository Structure](#Repository-Structure)
- [Dataset Format](#Dataset-Format)
- [Model Training and Inference](#Model-Training-and-Interface)
- [Visualization Outputs](#Visualization-Output)
- [Installation](#Installation)
- [Notes](#Notes)
- [Credits](#Credits)
- [Future Enhancements](#Future-Enhancements)

## Features

- Support for partial facial inputs (masked, blurred, occluded, etc.)
- Pretrained model using EfficientNetB0 (transfer learning)
- Model fine-tuning with learning rate scheduling and early stopping
- Metadata-driven identity mapping
- Top-3 prediction capability with confidence scores
- Confusion matrix and training visualizations
- Interactive GUI for image upload and real-time prediction using Streamlit

## Repository Structure

```
├── Main.py                        # Fine-tune an already trained model
├── pretrained_model.py           # Initial training script from scratch
├── predict.py                    # Script to predict identity using a file path
├── streamlit_app.py              # GUI application using Streamlit
├── confusion_matrix.png          # Confusion matrix after initial training
├── fine_tune_confusion_matrix.png # Confusion matrix after fine-tuning
├── fine_tune_accuracy.png        # Accuracy chart from fine-tuning
├── learning_curve_accuracy.png   # Accuracy chart from initial training
├── partial_face_model.keras      # Trained model file (not included in repo)
├── label_encoder.pkl             # Label encoder mapping identities to labels
├── metadata.csv                  # Image metadata and identity mapping
├── requirements.txt              # List of required Python packages
```

## Dataset Format

The dataset should be placed in a folder named `partial_face_dataset/` and include:

- Subfolders like `id_0001/`, `id_0002/`...
  - Each subfolder contains multiple facial image variants (e.g., `masked.jpg`, `top_crop.jpg`, etc.)

The `metadata.csv` should contain two columns:
- `filename`: relative path to the image inside `partial_face_dataset/`
- `identity`: corresponding class/label for the person

> Note: Due to size constraints, the dataset and model are not included in this repository. You must place them manually in the project root.

## Model Training and Inference

### Step 1: Train from Scratch
Train the base model using EfficientNet and save trained weights:
```bash
python pretrained_model.py
```

### Step 2: Fine-Tune the Model
Improve performance by continuing training with reduced learning rate:
```bash
python Main.py
```

### Step 3: Predict Using Image Path
Modify `predict.py` with a valid image path, for example:
```python
predict_image("partial_face_dataset/id_0005/masked.jpg")
```
Then run:
```bash
python predict.py
```

### Step 4: Use Streamlit Web App
Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```
Upload a partial face image and the app will display:
- Top-3 predicted identities with confidence scores
- Reference image of predicted identity

## Visualization Outputs

- `confusion_matrix.png`: Initial training evaluation
- `fine_tune_confusion_matrix.png`: Evaluation after fine-tuning
- `fine_tune_accuracy.png`: Accuracy chart during fine-tuning
- `learning_curve_accuracy.png`: Accuracy curve from base training

These images help understand how well the model is performing across training stages.

## Installation

Install all necessary packages using:
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install streamlit tensorflow opencv-python pandas matplotlib seaborn scikit-learn
```

## Notes

- Ensure that `partial_face_model.keras` and `label_encoder.pkl` exist in the project directory before inference.
- Dataset should match the structure described in `metadata.csv`.
- You can retrain or fine-tune by adding more identities and running the respective training scripts.

## Credits

This project was developed as a collaborative group assignment by:

- **Raja Ali Tabish**  
- **Aatma Ram**  
- **Moiz Kiani**  
- **Aparna Ghimire**

Supervised and guided as part of academic coursework in deep learning and AI applications.

## Future Enhancements

- Add support for face embedding comparisons
- Deploy Streamlit app on Streamlit Cloud or Hugging Face Spaces
- Add identity search history and user feedback tracking
- Integrate camera-based live recognition
- Improve dataset diversity and increase model robustness

For any feedback, contributions, or collaboration inquiries, feel free to open an issue or fork the repository.

