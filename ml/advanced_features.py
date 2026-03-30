import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# ================= CLAHE =================
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)


# ================= LOG GABOR + ATTENTION =================
def log_gabor_attention(img):

    orientations = [0, 45, 90, 135]

    # 🔥 IMPORTANT (faculty requirement)
    weights = {
        0: 0.5,
        45: 2.0,
        90: 2.0,
        135: 0.5
    }

    features = []

    for angle in orientations:

        kernel = cv2.getGaborKernel(
            (21,21),
            5,
            np.radians(angle),
            10,
            0.5,
            0
        )

        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)

        # apply attention
        filtered = filtered * weights[angle]

        features.append(np.mean(filtered))
        features.append(np.std(filtered))

    return features


# ================= LTP FEATURE =================
def ltp_feature(img):

    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(img, n_points, radius, method='uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=20, range=(0,20))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist.tolist()


# ================= FINAL FEATURE =================
def extract_features(img, use_ltp=True):

    img = apply_clahe(img)

    f1 = log_gabor_attention(img)

    if use_ltp:
        f2 = ltp_feature(img)
        return np.array(f1 + f2)
    else:
        return np.array(f1)