import numpy as np
import matplotlib.pyplot as plt
# Step 1: Generate a dataset with clusters based on 3 features
from sklearn.datasets import make_blobs

# Create a dataset with 3 clusters in 3D space
data_3d, labels = make_blobs(n_samples=300, centers=3, n_features=3, random_state=40, cluster_std=2)

# Step 2: Create two derived features that do not show the clusters
# We'll create new features as combinations of the original features
derived_feature_1 = data_3d[:, 0] * data_3d[:, 1]  # Feature 1 + Feature 2
derived_feature_2 = 2 * data_3d[:, 0] - 3 * data_3d[:, 2]  # Feature 1 - Feature 3

# Combine the derived features into a new dataset for the 2D plot
data_2d = np.vstack((derived_feature_1, derived_feature_2)).T

# Step 3: Plot the 2D plot using derived features (Feature 1 and Feature 2)
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', alpha=0.6)
plt.title('2D Scatter Plot (Derived Features - No Clear Clusters)')
#plt.xlabel('Derived Feature 1 (Feature 1 + Feature 2)')
#plt.ylabel('Derived Feature 2 (Feature 1 - Feature 3)')
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.grid()
plt.show()

# Step 4: Plot the 3D plot using the original features
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], alpha=0.6)
#ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', alpha=0.6)
#ax.set_title('3D Scatter Plot (Revealing Clear Clusters)')
#ax.set_xlabel('Feature 1')
#ax.set_ylabel('Feature 2')
#ax.set_zlabel('Feature 3')
plt.show()

derived_feature_1 = data_3d[:, 1]
derived_feature_2 = data_3d[:, 2]
data_2d = np.vstack((derived_feature_1, derived_feature_2)).T

# Step 3: Plot the 2D plot using derived features (Feature 1 and Feature 2)
plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', alpha=0.6)
plt.title('2D Scatter Plot (Derived Features - Clear Clusters)')
#plt.xlabel('Derived Feature 1 (Feature 1 + Feature 2)')
#plt.ylabel('Derived Feature 2 (Feature 1 - Feature 3)')
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.grid()
plt.show()