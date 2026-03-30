import numpy as np
import dtcwt

X_train = np.load("X_train_mc.npy")
X_test = np.load("X_test_mc.npy")

transform = dtcwt.Transform2d()

def extract_features(images, levels=3):
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

print("Extracting DTCWT features...")
X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)

np.save("X_train_mc_dtcwt.npy", X_train_feat)
np.save("X_test_mc_dtcwt.npy", X_test_feat)

print("✅ Multi-class DTCWT done")
