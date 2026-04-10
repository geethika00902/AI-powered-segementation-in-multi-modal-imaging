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
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-4

TRAIN_IMAGE_PATH = "data/train/images"
TRAIN_MASK_PATH  = "data/train/masks"

VAL_IMAGE_PATH   = "data/val/images"
VAL_MASK_PATH    = "data/val/masks"

# =============================
# DICE METRIC
# =============================
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

# =============================
# LOAD IMAGE & MASK
# =============================
def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image / 255.0
    return image

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

# =============================
# LOAD DATASET
# =============================
def load_dataset(image_path, mask_path):
    images = sorted(glob(os.path.join(image_path, "*.png")))
    masks  = sorted(glob(os.path.join(mask_path, "*.png")))

    if len(images) == 0 or len(masks) == 0:
        raise ValueError("Images or masks not found")

    return images, masks

# =============================
# DATA GENERATOR
# =============================
def data_generator(images, masks, batch_size=BATCH_SIZE):
    while True:
        for i in range(0, len(images), batch_size):
            x, y = [], []
            for img, msk in zip(images[i:i + batch_size], masks[i:i + batch_size]):
                x.append(load_image(img))
                y.append(load_mask(msk))
            yield np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

# =============================
# MAIN
# =============================
if __name__ == "__main__":

    print("Loading training data...")
    train_x, train_y = load_dataset(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH)

    print("Loading validation data...")
    valid_x, valid_y = load_dataset(VAL_IMAGE_PATH, VAL_MASK_PATH)

    print(f"Training samples   : {len(train_x)}")
    print(f"Validation samples : {len(valid_x)}")

    train_gen = data_generator(train_x, train_y)
    valid_gen = data_generator(valid_x, valid_y)

    print("Building UNET model...")
    model = build_unet((IMAGE_SIZE, IMAGE_SIZE, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", dice_coef]
    )

    print("\nStarting training...\n")

    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        steps_per_epoch=max(1, len(train_x)//BATCH_SIZE),
        validation_steps=max(1, len(valid_x)//BATCH_SIZE)
    )

    model.save("models/unet_model.h5")
    print("\nModel saved as models/unet_model.h5")