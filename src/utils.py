import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import SGD, Adam, Adagrad, RMSprop
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
from sam_torch import SAM

import numpy as np
import os
from typing import Tuple, List

# the default value for "physical batch size", which is the largest batch size that we try to put on the GPU
DEFAULT_PHYS_BS = 1000

def epoch_to_idx(epoch: int, batch_size: int, dataset_size: int) -> int:

    batchs_per_epoch = int(np.ceil(dataset_size / batch_size))
    idx = batchs_per_epoch * epoch

    return idx

def get_directory(
        dataset: str,
        arch_id: float,
        loss: float,
        seed: int,
        opt: str,
        lr: float,
        beta: float,
        rho: float,
        batch_size: int
    ) -> str:

    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/{loss}/seed_{seed}/{opt}"

    if opt == "sgd":
        if beta == 0:
            return f"{directory}/lr_{lr}_batch_{batch_size}"
        else:
            return f"{directory}/lr_{lr}_batch_{batch_size}_beta_{beta}"
    elif opt == "sam":
        return f"{directory}/lr_{lr}_batch_{batch_size}/rho_{rho}"
    elif opt == "adam":
        return f"{directory}/lr_{lr}_batch_{batch_size}_beta_{beta}"
    else:
        raise ValueError


def get_proj_directory(
        proj:str,
        dataset: str,
        arch_id: str,
        loss: str,
        seed: int,
        opt: str,
        lr: float,
        beta: float,
        rho: float,
        batch_size: int,
        start_step: int
    ):

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


def get_optimizer(parameters, opt: str, lr: float, beta: float, rho: float) -> Optimizer:
    
    if opt == "sgd":
        return SGD(parameters, lr=lr, momentum=beta, nesterov=False)
    elif opt == "sam":
        return SAM(parameters, SGD, rho=rho, adaptive=False, lr=lr, momentum=0)
    elif opt == "adam":
        return Adam(parameters, lr=lr, betas=(beta, 0.999))
    else:
        raise ValueError


def save_files(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}")


def save_files_final(directory: str, arrays: List[Tuple[str, torch.Tensor]]):
    """Save a bunch of tensors."""
    for (arr_name, arr) in arrays:
        torch.save(arr, f"{directory}/{arr_name}_final")


def iterate_dataset(dataset: Dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()


def compute_losses(network: nn.Module, loss_functions: List[nn.Module], dataset: Dataset,
                   batch_size: int = DEFAULT_PHYS_BS):
    """Compute loss over a dataset."""
    L = len(loss_functions)
    losses = [0. for l in range(L)]
    with torch.no_grad():
        for (X, y) in iterate_dataset(dataset, batch_size):
            preds = network(X)
            for l, loss_fn in enumerate(loss_functions):
                losses[l] += loss_fn(preds, y) / len(dataset)
    return losses


def get_loss_and_acc(loss: str):
    """
    Return modules to compute the loss and accuracy.  The loss module should be "sum" reduction.
    """
    if loss == "mse":
        return SquaredLoss(), SquaredAccuracy()
    elif loss == "ce":
        return nn.CrossEntropyLoss(reduction='sum'), AccuracyCE()
    raise NotImplementedError(f"no such loss function: {loss}")

def compute_gradient(network: nn.Module, loss_fn: nn.Module,
                     dataset: Dataset, physical_batch_size: int = DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at the current network parameters. """
    p = len(parameters_to_vector(network.parameters()))
    average_gradient = torch.zeros(p, device='cuda')
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        batch_loss = loss_fn(network(X), y) / len(dataset)
        batch_gradient = parameters_to_vector(
            torch.autograd.grad(batch_loss, inputs=network.parameters()))
        average_gradient += batch_gradient
    return average_gradient    

class AtParams(object):
    """ Within a with block, install a new set of parameters into a network.

    Usage:

        # suppose the network has parameter vector old_params
        with AtParams(network, new_params):
            # now network has parameter vector new_params
            do_stuff()
        # now the network once again has parameter vector new_params
    """

    def __init__(self, network: nn.Module, new_params: Tensor):
        self.network = network
        self.new_params = new_params

    def __enter__(self):
        self.stash = parameters_to_vector(self.network.parameters())
        vector_to_parameters(self.new_params, self.network.parameters())

    def __exit__(self, type, value, traceback):
        vector_to_parameters(self.stash, self.network.parameters())


def compute_gradient_at_theta(network: nn.Module, loss_fn: nn.Module, dataset: Dataset,
                              theta: torch.Tensor, batch_size=DEFAULT_PHYS_BS):
    """ Compute the gradient of the loss function at arbitrary network parameters "theta".  """
    with AtParams(network, theta):
        return compute_gradient(network, loss_fn, dataset, physical_batch_size=batch_size)

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