import numpy as np
import cv2
from skimage.filters import gabor

# Load CLAHE images
X_train = np.load("X_train_mc.npy")
X_test = np.load("X_test_mc.npy")

def extract_log_gabor_features(images):
    features = []

    # Optimized parameters
    frequency = 0.2
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for img in images:
        # 🔹 Downscale for speed
        img_small = cv2.resize(img, (64, 64))

        feat = []
        for theta in thetas:
            real, imag = gabor(img_small, frequency=frequency, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)

            feat.append(np.mean(magnitude))
            feat.append(np.std(magnitude))

        features.append(feat)

    return np.array(features)

print("🔹 Extracting FAST Log-Gabor features (training)...")
X_train_lg = extract_log_gabor_features(X_train)

print("🔹 Extracting FAST Log-Gabor features (testing)...")
X_test_lg = extract_log_gabor_features(X_test)

np.save("X_train_loggabor.npy", X_train_lg)
np.save("X_test_loggabor.npy", X_test_lg)

print("✅ Fast Log-Gabor feature extraction complete")
print("Training shape:", X_train_lg.shape)
print("Testing shape:", X_test_lg.shape)
