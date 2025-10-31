import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector
from sklearn.decomposition import PCA

from typing import Tuple

def init_weight_window(network: torch.nn.Module, window_size: int, device=None, dtype=torch.float32):

    n_params = sum(p.numel() for p in network.parameters())
    device = device or next(network.parameters()).device
    
    window = torch.zeros((n_params, window_size), dtype=dtype, device=device)
    return window, int(n_params)

def insert_into_window(window: torch.Tensor, vec: torch.Tensor, pointer: int):
    """
    Insert vec (shape [n_params]) into column `pointer` of window in-place.
    Returns next pointer (circular).
    """
    window[:, pointer] = vec
    pointer += 1
    if pointer >= window.shape[1]:
        pointer = 0
    return pointer

def get_ordered_window(window: torch.Tensor, pointer: int, filled: int):
    """
    Return a view of the window ordered chronologically:
      - if filled < window_size: returns first `filled` columns (oldest->newest).
      - else returns columns starting from pointer (oldest) to pointer - 1 (newest).
    `pointer` is the index of the next insertion (i.e. newest is pointer-1).
    """
    window_size = window.shape[1]
    if filled < window_size:
        return window[:, :filled].clone()  # shape (n_params, filled)
    # when full, oldest column is at `pointer`
    if pointer == 0:
        return window.clone()  # already oldest->newest
    return torch.cat([window[:, pointer:], window[:, :pointer]], dim=1).clone()

def get_pca_directions(
        weight_traj: Tensor,
        num_components: int,
        discard: bool=False
) -> Tuple[Tensor, Tensor]:
    
    weight_traj_window_np = weight_traj.numpy()
    
    pca = PCA(n_components=num_components)
    pca.fit(weight_traj_window_np)

    components = pca.components_
    variances = pca.explained_variance_

    components, variances = torch.from_numpy(components), torch.from_numpy(variances)

    # discard top variance
    if discard:
        components_diffusive = components[1:]
        variances_diffusive = variances[1:]
        return components_diffusive, variances_diffusive
    else:
        return components, variances