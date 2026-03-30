import numpy as np
from scipy.stats import ttest_ind

X_dtcwt = np.load("X_train_mc_dtcwt.npy")
X_hybrid = np.load("X_train_all.npy")

t_stat, p_val = ttest_ind(X_dtcwt.mean(axis=1), X_hybrid.mean(axis=1))
print("t-statistic:", t_stat)
print("p-value:", p_val)
