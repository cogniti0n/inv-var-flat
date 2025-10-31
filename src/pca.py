import torch
from torch import Tensor
from sklearn.decomposition import PCA

from typing import Tuple

def get_pca_directions(
        weight_traj: Tensor,
        num_components: int,
        cutoff_start_idx: int,
        cutoff_end_idx: int|None=None,
        discard: bool=False
) -> Tuple[Tensor, Tensor]:
    
    try:
        if cutoff_end_idx == None:
            weight_traj_window_np = weight_traj[cutoff_start_idx:].numpy()
        else:
            weight_traj_window_np = weight_traj[cutoff_start_idx: cutoff_end_idx].numpy()
    except IndexError:
        raise ValueError("Weight trajectory needs to contain more time steps")
    
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