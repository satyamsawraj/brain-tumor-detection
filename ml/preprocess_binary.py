import os
import cv2
import numpy as np

IMG_SIZE = 224
DATASET_PATH = "../dataset"

tumor_classes = ["glioma", "meningioma", "pituitary"]
no_tumor_class = "notumor"

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def load_data(split):
    X, y = [], []

    for cls in tumor_classes:
        folder = os.path.join(DATASET_PATH, split, cls)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                X.append(preprocess_image(img_path))
                y.append(1)
            except:
                pass

    folder = os.path.join(DATASET_PATH, split, no_tumor_class)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            X.append(preprocess_image(img_path))
            y.append(0)
        except:
            pass

    return np.array(X), np.array(y)

print("🔹 Loading training data...")
X_train, y_train = load_data("Training")

print("🔹 Loading testing data...")
X_test, y_test = load_data("Testing")

print("✅ Training data shape:", X_train.shape, y_train.shape)
print("✅ Testing data shape:", X_test.shape, y_test.shape)
