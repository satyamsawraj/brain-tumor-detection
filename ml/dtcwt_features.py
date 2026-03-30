import numpy as np
import dtcwt

# Load CLAHE-processed data
X_train = np.load("X_train_clahe.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test_clahe.npy")
y_test = np.load("y_test.npy")

transform = dtcwt.Transform2d()

def extract_dtcwt_features(images, levels=3):
    features = []

    for img in images:
        img = img.astype(np.float32)
        coeffs = transform.forward(img, nlevels=levels)

        feat = []
        for level in coeffs.highpasses:
            magnitude = np.abs(level)
            feat.append(np.mean(magnitude))
            feat.append(np.std(magnitude))

        features.append(feat)

    return np.array(features)

print("🔹 Extracting DTCWT features from training data...")
X_train_feat = extract_dtcwt_features(X_train)

print("🔹 Extracting DTCWT features from testing data...")
X_test_feat = extract_dtcwt_features(X_test)

# Save features
np.save("X_train_dtcwt.npy", X_train_feat)
np.save("X_test_dtcwt.npy", X_test_feat)

print("✅ DTCWT feature extraction complete")
print("Training features shape:", X_train_feat.shape)
print("Testing features shape:", X_test_feat.shape)
