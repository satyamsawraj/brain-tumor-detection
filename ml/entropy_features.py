import numpy as np
from skimage.measure import shannon_entropy

# Load CLAHE-processed multiclass images
X_train = np.load("X_train_mc.npy")
X_test = np.load("X_test_mc.npy")

def extract_entropy(images):
    ent = []
    for img in images:
        ent.append(shannon_entropy(img))
    return np.array(ent).reshape(-1, 1)

print("🔹 Extracting entropy features (training)...")
X_train_entropy = extract_entropy(X_train)

print("🔹 Extracting entropy features (testing)...")
X_test_entropy = extract_entropy(X_test)

np.save("X_train_entropy.npy", X_train_entropy)
np.save("X_test_entropy.npy", X_test_entropy)

print("✅ Entropy feature extraction complete")
print("Training shape:", X_train_entropy.shape)
print("Testing shape:", X_test_entropy.shape)
