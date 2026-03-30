import os
import cv2
import numpy as np

IMG_SIZE = 224
DATASET_PATH = "../dataset"

tumor_classes = ["glioma", "meningioma", "pituitary"]
no_tumor_class = "notumor"

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image_clahe(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = clahe.apply(img)
    img = img / 255.0
    return img

def load_data(split):
    X, y = [], []

    # tumor images
    for cls in tumor_classes:
        folder = os.path.join(DATASET_PATH, split, cls)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                X.append(preprocess_image_clahe(img_path))
                y.append(1)
            except:
                pass

    # no tumor images
    folder = os.path.join(DATASET_PATH, split, no_tumor_class)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            X.append(preprocess_image_clahe(img_path))
            y.append(0)
        except:
            pass

    return np.array(X), np.array(y)

print("🔹 Processing training data with CLAHE...")
X_train, y_train = load_data("Training")

print("🔹 Processing testing data with CLAHE...")
X_test, y_test = load_data("Testing")

# Save to disk
np.save("X_train_clahe.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test_clahe.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Saved CLAHE-processed datasets")
print("Training shape:", X_train.shape, y_train.shape)
print("Testing shape:", X_test.shape, y_test.shape)
