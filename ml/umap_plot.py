import numpy as np
import umap
import matplotlib.pyplot as plt

X = np.load("X_train_all.npy")
y = np.load("y_train_mc.npy")

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=y, cmap="tab10", s=5)
plt.title("UMAP Projection of Hybrid Features")
plt.colorbar(label="Tumor Class")
plt.show()
# local ternary pattern