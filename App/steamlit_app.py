import streamlit as st
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Configuration:
MODEL_PATH = "partial_face_model.keras"
ENCODER_PATH = "label_encoder.pkl"
DATASET_DIR = "partial_face_dataset"
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
IMG_SIZE = 128

# Loading model and encoder:
@st.cache_resource
def load_model_and_encoder():
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# Load metadata.csv
@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_PATH)

df = load_metadata()

# Run Prediction Function:
def predict_image(image):
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_input = np.expand_dims(img_preprocessed, axis=0)
    predictions = model.predict(img_input)[0]
    top3_idx = predictions.argsort()[-3:][::-1]
    top3_labels = label_encoder.inverse_transform(top3_idx)
    top3_conf = predictions[top3_idx]
    return top3_labels, top3_conf

# Interface of our Application for displaying on web browser:
st.title("Partial Face Recognition App")
st.write("Upload a partial face image to predict the person's identity.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the input image:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    # Prediction:
    top3_labels, top3_conf = predict_image(image)

    st.subheader("Top 3 Predictions:")
    for i in range(3):
        st.write(f"{i+1}. {top3_labels[i]} ({top3_conf[i]*100:.2f}%)")

    # Show reference image in result for matching the output:
    predicted_class = top3_labels[0]
    ref_row = df[df['identity'] == predicted_class].iloc[0]
    ref_path = os.path.join(DATASET_DIR, ref_row['filename'])
    ref_img = cv2.imread(ref_path)
    if ref_img is not None:
        st.subheader("Reference Image from Training Data:")
        st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), caption=f"{predicted_class}", use_column_width=True)
