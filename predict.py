import os
import cv2
import numpy as np
from glob import glob
from unet import build_unet

IMAGE_SIZE = 128          # same as training
MODEL_PATH = "unet_model.h5"
IMAGE_DIR = "data/images"
SAVE_DIR = "results"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading model...")
model = build_unet((IMAGE_SIZE, IMAGE_SIZE, 3))
model.load_weights(MODEL_PATH)

print("Running predictions...")
images = glob(os.path.join(IMAGE_DIR, "*.png"))

for img_path in images:   # first 10 images
    name = os.path.basename(img_path)

    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)[0]
    pred = (pred > 0.2).astype("uint8") * 255


    cv2.imwrite(os.path.join(SAVE_DIR, name), pred)

print("✅ Predictions saved in UNET/results/")
