import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# === INPUT/OUTPUT PATHS ===
input_dir = "part3"  # Change this to your actual folder name
output_dir = "partial_face_dataset"
os.makedirs(output_dir, exist_ok=True)

metadata = []

# === TRANSFORMATION FUNCTIONS ===
def crop_top(img):
    return img[:img.shape[0]//2, :]

def crop_left(img):
    return img[:, :img.shape[1]//2]

def blur_face(img):
    return cv2.GaussianBlur(img, (31, 31), 0)

def partial_blur(img):
    h, w = img.shape[:2]
    output = img.copy()
    output[h//5:h//3, w//4:3*w//4] = cv2.GaussianBlur(output[h//5:h//3, w//4:3*w//4], (21,21), 0)
    return output

def add_synthetic_mask(img):
    h, w = img.shape[:2]
    mask_img = img.copy()
    mask_color = (50, 50, 50)
    mask_top = int(0.55 * h)
    mask_bottom = int(0.75 * h)
    cv2.rectangle(mask_img, (int(0.2 * w), mask_top), (int(0.8 * w), mask_bottom), mask_color, -1)
    return mask_img

def add_random_box(img):
    h, w = img.shape[:2]
    box_img = img.copy()
    x1 = np.random.randint(0, w//2)
    y1 = np.random.randint(0, h//2)
    x2 = x1 + np.random.randint(20, 50)
    y2 = y1 + np.random.randint(20, 50)
    cv2.rectangle(box_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return box_img

def add_noise_patch(img):
    h, w = img.shape[:2]
    noisy_img = img.copy()
    x = np.random.randint(0, w - 40)
    y = np.random.randint(0, h - 40)
    noise = np.random.randint(0, 255, (40, 40, 3), dtype='uint8')
    noisy_img[y:y+40, x:x+40] = noise
    return noisy_img

# === PROCESSING IMAGES ===
image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]

for i, filename in enumerate(tqdm(image_files[:1100])):  # Adjust limit if needed
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (128, 128))
    base_name = f"id_{i:04d}"
    person_dir = os.path.join(output_dir, base_name)
    os.makedirs(person_dir, exist_ok=True)

    # Variants
    variants = {
        "original": img,
        "top_crop": crop_top(img),
        "left_crop": crop_left(img),
        "blurred": blur_face(img),
        "partial_blur": partial_blur(img),
        "synthetic_mask": add_synthetic_mask(img),
        "occlusion_box": add_random_box(img),
        "noise_patch": add_noise_patch(img),
    }

    for var_name, var_img in variants.items():
        out_path = f"{person_dir}/{var_name}.jpg"
        cv2.imwrite(out_path, var_img)
        metadata.append([out_path, base_name, var_name])

# === SAVE METADATA CSV ===
df = pd.DataFrame(metadata, columns=["filename", "identity", "transformation"])
df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print("✅ Partial face dataset created successfully!")
