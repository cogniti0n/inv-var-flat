import numpy as np
from torchvision.datasets import CIFAR10
from typing import Tuple
from torch.utils.data.dataset import TensorDataset
import os
import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Tuple

DATASETS_FOLDER = os.environ["DATASETS"]

def center(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape: Tuple) -> np.ndarray:
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor: Tensor, num_classes: int, default=0) -> Tensor:
    
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y: Tensor, loss: str) -> Tensor:

    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)
    else:
        raise ValueError("Loss must be either {'ce', 'mse'}")

def load_cifar(loss: str) -> Tuple[TensorDataset, TensorDataset]:

    cifar10_train = CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = CIFAR10(root=DATASETS_FOLDER, download=True, train=False)

    X_train = flatten(cifar10_train.data / 255)
    X_test = flatten(cifar10_test.data / 255)

    y_train = make_labels(torch.tensor(cifar10_train.targets), loss)
    y_test = make_labels(torch.tensor(cifar10_test.targets), loss)
    
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)

    train_images = unflatten(standardized_X_train, (32, 32, 3))
    train_images = torch.from_numpy(train_images.transpose((0, 3, 1, 2))).float()
    train = TensorDataset(train_images, y_train)

    test_images = unflatten(standardized_X_test, (32, 32, 3))
    test_images = torch.from_numpy(test_images.transpose((0, 3, 1, 2))).float()
    test = TensorDataset(test_images, y_test)

    return train, test