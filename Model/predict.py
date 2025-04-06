import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# === CONFIG ===
MODEL_PATH = "partial_face_model.keras"
ENCODER_PATH = "label_encoder.pkl"
DATASET_DIR = "partial_face_dataset"
IMG_SIZE = 128

# === Load model and encoder ===
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# === Load metadata for reference image ===
metadata_path = os.path.join(DATASET_DIR, 'metadata.csv')
df = pd.read_csv(metadata_path)

# === Prediction Function ===
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image.")
        return

    # Resize and preprocess
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_input = np.expand_dims(img_input, axis=0)

    # Make prediction
    prediction = model.predict(img_input)[0]
    top3_idx = prediction.argsort()[-3:][::-1]

    print("\nTop-3 Predictions:")
    for i, idx in enumerate(top3_idx):
        label = label_encoder.inverse_transform([idx])[0]
        confidence = prediction[idx] * 100
        print(f"{i+1}. {label} ({confidence:.2f}%)")

    # Display result
    predicted_class = label_encoder.inverse_transform([top3_idx[0]])[0]
    ref_row = df[df['identity'] == predicted_class].iloc[0]
    ref_img_path = os.path.join(DATASET_DIR, ref_row['filename'])
    ref_img = cv2.imread(ref_img_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if ref_img is not None:
        plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Top-1 Prediction: {predicted_class}")
    else:
        plt.title("Reference Not Found")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.close()
    print("Prediction image saved as prediction_result.png")

# === USAGE ===
# Replace this path with an image you want to test
predict_image("partial_face_dataset/id_0150/top_crop.jpg")
