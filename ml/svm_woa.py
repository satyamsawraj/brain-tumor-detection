import numpy as np
import cv2
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from advanced_features import extract_features

# =========================
# SETTINGS
# =========================
np.random.seed(42)

DATASET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "Training")
)

IMG_SIZE = 224
CLASSES = ["notumor", "glioma", "meningioma", "pituitary"]

# =========================
# LOAD DATASET
# =========================
def load_dataset():
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_DIR, cls)

        if not os.path.exists(class_dir):
            print(f"⚠️ Missing folder: {class_dir}")
            continue

        for i, file in enumerate(os.listdir(class_dir)):
            if i % 50 == 0:
                print(f"Processing {cls}: {i}")

            img_path = os.path.join(class_dir, file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # 🔥 FIXED FEATURE EXTRACTION
            features = np.array(extract_features(img)[0]).flatten()
            features = extract_features(img, use_ltp=True)
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("DEBUG SHAPE:", X.shape)

    return X, y


print("🔄 Loading dataset...")
X, y = load_dataset()

print(f"✅ Samples: {len(X)}, Features: {X.shape[1]}")

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# WOA PARAMETERS
# =========================
num_whales = 12
iterations = 20

whales = np.random.rand(num_whales, 2)

# =========================
# LOG SCALE PARAMS
# =========================
def scale_params(whale):
    C = 10 ** (whale[0] * 3)        # 1 → 1000
    gamma = 10 ** (-3 + whale[1]*3) # 0.001 → 1
    return C, gamma

# =========================
# FITNESS FUNCTION
# =========================
def fitness(whale):
    try:
        C, gamma = scale_params(whale)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced"))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=2)

        return cross_val_score(model, X_train, y_train, cv=2, scoring="f1_weighted").mean()

    except:
        return 0

# =========================
# WOA OPTIMIZATION
# =========================
best_whale = whales[0]
best_score = -1

print("🐋 Running WOA optimization...")

for iter in range(iterations):
    a = 2 - iter * (2 / iterations)

    for i in range(num_whales):
        score = fitness(whales[i])

        if score > best_score:
            best_score = score
            best_whale = whales[i]

    for i in range(num_whales):
        r1 = np.random.rand()
        r2 = np.random.rand()

        A = 2 * a * r1 - a
        C_rand = 2 * r2
        p = np.random.rand()

        if p < 0.5:
            D = abs(C_rand * best_whale - whales[i])
            whales[i] = best_whale - A * D
        else:
            D = abs(best_whale - whales[i])
            l = np.random.uniform(-1, 1)

            # 🔥 FIXED SPIRAL EQUATION
            b=1
            whales[i] = D * np.exp(b * l) * np.cos(2*np.pi*l) + best_whale

        whales[i] = np.clip(whales[i], 0, 1)

    print(f"Iteration {iter+1} → Best Accuracy: {best_score:.4f}")

# =========================
# FINAL MODEL
# =========================
best_C, best_gamma = scale_params(best_whale)

print("\n🔥 Best Params:")
print("C =", best_C)
print("gamma =", best_gamma)

final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(C=best_C, gamma=best_gamma, kernel="rbf", probability=True))
])

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\n✅ FINAL RESULT (WOA-SVM)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
joblib.dump(final_model, "svm_woa_model.pkl")
print("💾 WOA optimized model saved!")