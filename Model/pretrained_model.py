import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import gc
import pickle

# Configuration:
DATASET_DIR = 'partial_face_dataset'
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 1000
MODEL_PATH = "partial_face_model.keras"
ENCODER_PATH = "label_encoder.pkl"

# SETUP:
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# Process no.1: Load Metadata:
metadata_path = os.path.join(DATASET_DIR, 'metadata.csv')
df = pd.read_csv(metadata_path)

# Process no.2: Train and Test Split:
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['identity'], random_state=42)

# Process no.3: Image Generators:
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=DATASET_DIR,
    x_col='filename',
    y_col='identity',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=DATASET_DIR,
    x_col='filename',
    y_col='identity',
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Process no.4: Label Encoder Save:
label_encoder = LabelEncoder()
label_encoder.fit(df['identity'])

with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)

# Process no.5: Build Model (EfficientNetB0 + Custom Head):
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)
model.summary()

# Process no.6: Callbacks:
def lr_schedule(epoch):
    if epoch < 5:
        return 1e-3
    elif epoch < 15:
        return 5e-4
    else:
        return 1e-4

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    LearningRateScheduler(lr_schedule)
]

# Process no.7A: Train Model (Initial):
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Process no.7B: Fine-Tune:
print("\nUnfreezing top layers for fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

history_fine = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=callbacks
)

# Process no.8: Evaluate Model
loss, acc, top3_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")

# Process no.9A: Confusion Matrix:
print("Predicting test set...")
y_pred = model.predict(test_generator, verbose=1)
print("Prediction complete.")
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# Process no.9B: Per-Type Evaluation:
test_df['predicted'] = [label_encoder.classes_[i] for i in y_pred_classes]
test_df['true'] = [label_encoder.classes_[i] for i in y_true]
test_df['type'] = test_df['filename'].apply(lambda x: os.path.basename(x).split('.')[0])
print("\n--- Accuracy by Partial Face Type ---")
for face_type in test_df['type'].unique():
    subset = test_df[test_df['type'] == face_type]
    correct = np.sum(subset['predicted'] == subset['true'])
    acc = correct / len(subset)
    print(f"{face_type}: {acc*100:.2f}% accuracy")

# Process no.9C: Learning Curves:
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
if 'accuracy' in history_fine.history:
    plt.plot(history_fine.history['accuracy'], label='Fine-tuned Train')
    plt.plot(history_fine.history['val_accuracy'], label='Fine-tuned Val')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("learning_curve_accuracy.png")
plt.close()

# Process no.10: Save Model:
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Process no.11: Predict Image with Top-3 Reference:
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image.")
        return

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_input = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_input = np.expand_dims(img_input, axis=0)
    prediction = model.predict(img_input)[0]
    top3_idx = prediction.argsort()[-3:][::-1]

    print("\nTop-3 Predictions:")
    for i, idx in enumerate(top3_idx):
        label = label_encoder.inverse_transform([idx])[0]
        print(f"{i+1}. {label} ({prediction[idx]*100:.2f}%)")

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
