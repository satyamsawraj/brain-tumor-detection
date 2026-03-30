import numpy as np

# Load features
X_train_dtcwt = np.load("X_train_mc_dtcwt.npy")
X_test_dtcwt = np.load("X_test_mc_dtcwt.npy")

X_train_lg = np.load("X_train_loggabor.npy")
X_test_lg = np.load("X_test_loggabor.npy")

# Combine (DTCWT + Log-Gabor)
X_train_final = np.hstack((X_train_dtcwt, X_train_lg))
X_test_final = np.hstack((X_test_dtcwt, X_test_lg))

# Save
np.save("X_train_final.npy", X_train_final)
np.save("X_test_final.npy", X_test_final)

print("✅ Feature fusion complete")
print("Training shape:", X_train_final.shape)
print("Testing shape:", X_test_final.shape)
