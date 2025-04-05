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
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import gc
import pickle

# Configuration of the Model.
DATASET_DIR = 'partial_face_dataset'
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20  # Fine-tuning epochs.
MODEL_PATH = "partial_face_model.keras"
ENCODER_PATH = "label_encoder.pkl"

# Model Setup.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# Process no.1: Loading Metadata.csv.
metadata_path = os.path.join(DATASET_DIR, 'metadata.csv')
df = pd.read_csv(metadata_path)

# Process no. 2: Train and Test Split.
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['identity'], random_state=42)

# Process no. 3: Image formation.
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

# Process no. 4: Loading the label encoder.
with open(ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Process no. 5: Loading of the Existing Model as we have run the pretrained_model.py earlier and Model is already trained.
if os.path.exists(MODEL_PATH):
    print("Loading existing trained model for continued training...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    base_model = model.layers[0]  #  Using EfficientNetB0 base.
else:
    raise FileNotFoundError("No pre-trained model found to continue training.")

# Process no. 6: Unfreezing the top layers for fine-tuning.
print("Unfreezing top layers for fine-tuning...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# === STEP 7: Compile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

# === STEP 8: Callbacks and Continue Training ===
def lr_schedule(epoch):
    if epoch < 5:
        return 1e-5
    elif epoch < 10:
        return 5e-6
    else:
        return 1e-6

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    LearningRateScheduler(lr_schedule)
]

print("Continuing fine-tuning...")
history_fine = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === STEP 9: Save Improved Model ===
model.save(MODEL_PATH)
print(f"Updated model saved to {MODEL_PATH}")

# === STEP 10: Accuracy Metrics ===
eval_loss, eval_acc, eval_top3 = model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {eval_acc * 100:.2f}%")
print(f"Final Top-3 Accuracy: {eval_top3 * 100:.2f}%")

# === STEP 11: Confusion Matrix & Classification Report ===
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("fine_tune_confusion_matrix.png")
plt.close()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# === STEP 12: Per-Type Accuracy ===
test_df['predicted'] = [label_encoder.classes_[i] for i in y_pred]
test_df['true'] = [label_encoder.classes_[i] for i in y_true]
test_df['type'] = test_df['filename'].apply(lambda x: os.path.basename(x).split('.')[0])
print("\n--- Accuracy by Partial Face Type ---")
for face_type in test_df['type'].unique():
    subset = test_df[test_df['type'] == face_type]
    correct = np.sum(subset['predicted'] == subset['true'])
    acc = correct / len(subset)
    print(f"{face_type}: {acc*100:.2f}% accuracy")

# === STEP 13: Learning Curve Plot ===
plt.plot(history_fine.history['accuracy'], label='Fine-tuned Train')
plt.plot(history_fine.history['val_accuracy'], label='Fine-tuned Val')
plt.title('Fine-Tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("fine_tune_accuracy.png")
plt.close()

# === STEP 14: RAM Cleanup ===
from tensorflow.keras import backend as K
K.clear_session()
gc.collect()
