import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

import argparse

from dataload.data import DATASETS
from utils import get_proj_directory

def main(
        proj: str|None=None,
        dataset: str="mnist",
        arch_id: str|None="fc",
        loss: str|None=None,
        opt: str|None=None,
        lr: float|None=None,
        max_steps: int|None=None,
        start_step: int|None=None,
):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the neural network by projecting onto PCA directions")

    parser.add_argument("proj", type=str, choices=["dom", "bulk"],
                        help="which subspace to project")
    parser.add_argument("dataset", type=str, choices=DATASETS,
                        help="which dataset to train")
    parser.add_argument("arch_id", type=str,
                        help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse", "logtanh"],
                        help="which loss function to use")
    parser.add_argument("opt", type=str, choices=["sgd", "sam", "adam"],
                        help="which optimization method to use")
    parser.add_argument("lr", type=float,
                        help="the learning rate")
    parser.add_argument("--max_steps", type=int,
                        help="the maximum number of gradient steps to train for", default=1000)
    parser.add_argument("--start_step", type=int,
                        help="the step to start projected method", default=0)
    parser.add_argument("--seed", type=int,
                        help="the random seed used when initializing the network weights", default=0)
    parser.add_argument("--beta", type=float,
                        help="momentum parameter", default=0)
    parser.add_argument("--rho", type=float,
                        help="perturbation radius for SAM", default=0)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--batch_size", type=int,
                        help="batch size of SGD", default=50)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value", default=1)
    parser.add_argument("--loss_goal", type=float,
                        help="terminate training if the train loss ever crosses this value", default=0)
    parser.add_argument("--neigs", type=int,
                        help="the number of top eigenvalues to compute")
    parser.add_argument("--neigs_dom", type=int,
                        help="the number of dominant top eigenvalues")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save results")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--gpu_id", type=int,
                        help="gpu (cuda device) id", default=0)
    
    args = parser.parse_args()
