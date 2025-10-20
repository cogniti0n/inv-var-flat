import numpy as np
from sklearn.decomposition import PCA
import os

from configs import CONFIGS

# TODO: fix time window (trucate first ~ 30 epochs)

N_COMPONENTS = 500

print("Loading weight trajectory...")
try:
    weight_trajectory = np.load('data/weight_trajectory.npy')
except FileNotFoundError:
    print("Error: 'data/weight_trajectory.npy' not found.")
    print("Please run 'train.py' first to generate the data.")
    exit()


print(f"Weight data shape: {weight_trajectory.shape}")

mean_weight = np.mean(weight_trajectory, axis=0)
np.save('data/mean_weight.npy', mean_weight)
print(f"Saved mean weight vector of shape: {mean_weight.shape}")


print(f"\nPerforming PCA for {N_COMPONENTS} components...")
pca = PCA(n_components=N_COMPONENTS)
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
print("Script 'run_pca.py' finished successfully.")
