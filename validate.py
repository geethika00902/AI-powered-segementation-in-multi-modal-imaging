import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from unet import build_unet

# =============================
# CONFIG
# =============================
IMAGE_SIZE = 256
MODEL_PATH = "models/unet_model.h5"

VAL_IMAGE_PATH = "data/val/images"
VAL_MASK_PATH  = "data/val/masks"

# =============================
# METRICS
# =============================
def dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# =============================
# LOAD IMAGE & MASK
# =============================
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return img

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

# =============================
# MAIN
# =============================
if __name__ == "__main__":

    print("Loading validation dataset...")

    val_images = sorted(glob(os.path.join(VAL_IMAGE_PATH, "*.png")))
    val_masks  = sorted(glob(os.path.join(VAL_MASK_PATH, "*.png")))

    if len(val_images) == 0:
        print("Validation images not found")
        exit()

    print(f"Total validation images: {len(val_images)}")

    print("Loading trained model...")
    model = build_unet((IMAGE_SIZE, IMAGE_SIZE, 3))
    model.load_weights(MODEL_PATH)

    total_dice = []
    total_iou = []
    tumor_count = 0
    non_tumor_count = 0

    print("Running validation...\n")

    for img_path, mask_path in zip(val_images, val_masks):

        image = load_image(img_path)
        mask = load_mask(mask_path)

        pred = model.predict(np.expand_dims(image, axis=0))[0]
        pred = (pred > 0.5).astype(np.uint8)

        d = dice_coef(mask, pred)
        i = iou_score(mask, pred)

        total_dice.append(d)
        total_iou.append(i)

        if np.any(pred):
            tumor_count += 1
        else:
            non_tumor_count += 1

    avg_dice = np.mean(total_dice)
    avg_iou = np.mean(total_iou)

    print("===================================")
    print(" VALIDATION RESULTS ")
    print("===================================")
    print(f"Total Images       : {len(val_images)}")
    print(f"Tumor Images       : {tumor_count}")
    print(f"Non-Tumor Images   : {non_tumor_count}")
    print(f"Average Dice Score: {avg_dice*100:.2f}%")
    print(f"Average IoU Score : {avg_iou*100:.2f}%")