import os
import cv2
import numpy as np
from tqdm import tqdm


IMG_SIZE = 224

# CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image_with_roi(img_path):
    """
    Reads image, applies segmentation + masking,
    then CLAHE enhancement on ROI.
    """

    # Read image
    img = cv2.imread(img_path)

    if img is None:
        return None

 

    # Resize ROI
    roi_img = cv2.resize(roi_img, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE ONLY on ROI
    roi_img = clahe.apply(roi_img)

    # Normalize
    roi_img = roi_img / 255.0

    return roi_img.astype(np.float32)


def load_dataset(base_dir, classes):
    """
    Loads dataset using ROI-based preprocessing
    """

    X = []
    y = []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)

        print(f"[INFO] Processing class: {class_name}")

        for img_name in tqdm(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)

            roi_img = preprocess_image_with_roi(img_path)

            if roi_img is None:
                continue

            X.append(roi_img)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":

    DATASET_DIR = "../dataset/Training"
    CLASSES = ["no_tumor", "glioma", "meningioma", "pituitary"]

    X, y = load_dataset(DATASET_DIR, CLASSES)

    print("[INFO] Dataset loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save for later stages
    np.save("X_train_mc_roi.npy", X)
    np.save("y_train_mc.npy", y)
