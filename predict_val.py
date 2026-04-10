import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from unet import build_unet

# =============================
# CONFIG
# =============================
IMAGE_SIZE = 256
MODEL_PATH = "unet_model.h5"

IMAGE_DIR = "data/val/images"
SAVE_DIR = "predictions"

os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# LOAD MODEL
# =============================
print("Building model...")
model = build_unet((IMAGE_SIZE, IMAGE_SIZE, 3))
model.load_weights(MODEL_PATH)
print("Model loaded successfully")

# =============================
# LOAD IMAGES
# =============================
images = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
print(f"Validation images found: {len(images)}")

# =============================
# PREDICT
# =============================
for img_path in images:
    name = os.path.basename(img_path)

    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image_norm = image / 255.0
    image_input = np.expand_dims(image_norm, axis=0)

    pred = model.predict(image_input, verbose=0)[0]
    pred = (pred > 0.5).astype(np.uint8) * 255

    save_path = os.path.join(SAVE_DIR, name)
    cv2.imwrite(save_path, pred)

print("✅ All validation predictions saved in 'predictions/'")
