import numpy as np

# Load individual feature sets
dtcwt_train = np.load("X_train_dtcwt.npy")
gabor_train = np.load("X_train_loggabor.npy")
entropy_train = np.load("X_train_entropy.npy")

dtcwt_test = np.load("X_test_dtcwt.npy")
gabor_test = np.load("X_test_loggabor.npy")
entropy_test = np.load("X_test_entropy.npy")

# Combine all features
X_train = np.hstack((dtcwt_train, gabor_train, entropy_train))
X_test  = np.hstack((dtcwt_test, gabor_test, entropy_test))

# Save final combined features
np.save("X_train_final.npy", X_train)
np.save("X_test_final.npy", X_test)

print("✅ FINAL feature set ready")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)