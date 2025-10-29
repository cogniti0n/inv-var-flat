import torch
from torch import Tensor
import torch.nn as nn

import numpy as np
import os
from typing import Tuple

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000

def epoch_to_idx(epoch: int, batch_size: int, dataset_size: int) -> int:

    batchs_per_epoch = int(np.ceil(dataset_size / batch_size))
    idx = batchs_per_epoch * epoch

    return idx

def get_loss_and_acc(loss: str) -> Tuple[nn.Module, nn.Module]:
    """
    Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction.
    """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")

def get_proj_directory(
        proj:str,
        dataset: str,
        arch_id: float,
        loss: float,
        seed: int,
        opt: str,
        lr: float,
        beta: float,
        rho: float,
        batch_size: int,
        start_step: int
    ) -> str:

    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/{loss}/seed_{seed}/{opt}"

    if opt == "sgd":
        if beta == 0:
            return f"{directory}/lr_{lr}_batch_{batch_size}/{proj}_{start_step}"
        else:
            return f"{directory}/lr_{lr}_batch_{batch_size}_beta_{beta}/{proj}_{start_step}"
    elif opt == "sam":
        return f"{directory}/lr_{lr}_batch_{batch_size}/rho_{rho}/{proj}_{start_step}"
    elif opt == "adam":
        return f"{directory}/lr_{lr}_batch_{batch_size}_beta_{beta}/{proj}_{start_step}"
    else:
        raise ValueError("'opt' variable must be in {'sgd', 'sam', 'adam'}")
    

class SquaredLoss(nn.Module):
    def forward(self, input: Tensor, target: Tensor):
        return 0.5 * ((input - target) ** 2).sum()

class SquaredAccuracy(nn.Module):
    def __init__(self):
        super(SquaredAccuracy, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target.argmax(1)).float().sum()

class AccuracyCE(nn.Module):
    def __init__(self):
        super(AccuracyCE, self).__init__()

    def forward(self, input, target):
        return (input.argmax(1) == target).float().sum()

class VoidLoss(nn.Module):
    def forward(self, X, Y):
        return 0