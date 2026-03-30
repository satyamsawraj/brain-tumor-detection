import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from advanced_features import extract_features

DATASET_DIR = "../dataset/Training"
IMG_SIZE = 224
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def load_dataset():
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, cls)

        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # 🔥 NEW FEATURES (IMPORTANT)
            features = extract_features(img, use_ltp=True)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("🔄 Loading dataset with Advanced Features...")
    X, y = load_dataset()

    print(f"✅ Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # Train SVM
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma = "scale", probability=True))
    ])
    model.fit(X, y)

    # Evaluate
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(
        y_test, y_pred,
        target_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    ))

    # Save model
    joblib.dump(model, "brain_tumor_advanced_model.pkl")
    print("💾 Advanced model saved!")