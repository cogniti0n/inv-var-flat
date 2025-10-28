import numpy as np
from sklearn.decomposition import PCA
import os

from configs import CONFIGS

num_components = CONFIGS["num_components"]
cutoff_epoch = CONFIGS["cutoff_epoch"]

print("Loading weight trajectory...")
try:
    weight_trajectory = np.load('data/weight_trajectory.npy')
    batch_size = int(np.load('data/batches_per_epoch.npy')[0])
    assert batch_size == CONFIGS["batch_size"]
except FileNotFoundError:
    raise FileNotFoundError("'data/weight_trajectory.npy' not found.")

print(f"Weight data shape: {weight_trajectory.shape}")

cutoff_idx = batch_size * cutoff_epoch
if cutoff_idx >= len(weight_trajectory):
    raise ValueError("Epochs to cut off is too large")
exploration_trajectory = weight_trajectory[cutoff_idx:]

mean_weight = np.mean(weight_trajectory, axis=0)
np.save('data/mean_weight.npy', mean_weight)
print(f"Saved mean weight vector of shape: {mean_weight.shape}")

print(f"\nPerforming PCA for {num_components} components...")
pca = PCA(n_components=num_components)
pca.fit(weight_trajectory)

components = pca.components_
variances = pca.explained_variance_

if not os.path.exists('data'):
    os.makedirs('data')

np.save('data/pca_components.npy', components)
np.save('data/pca_variances.npy', variances)

print("PCA analysis complete.")
print(f"Saved principal components of shape: {components.shape}")
print(f"Saved variances of shape: {variances.shape}")