from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import random
import numpy as np

class BaseSelector(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def select_indices(self, layer: nn.Conv2d) -> list:
        pass
    


class PuRLSelector(BaseSelector):
    def __init__(self) -> None:
        super().__init__()

    def select_indices(self, layer: nn.Conv2d, alpha: float) -> list:

        # Calculate norm for every channel
        norms = (torch.norm(layer.weight.data, 'fro', dim=[2, 3]))
        norms = torch.norm(norms, 'fro', dim=1)

        # Calculate the mean and standard deviation of the channel norms in the layer
        norms_avg = torch.mean(norms)
        norms_std = torch.std(norms)

        # Find the indices of the channels that have to be pruned
        norm_condition = torch.logical_and((torch.absolute(norms) < norms_avg + alpha * norms_std),
                                        (torch.absolute(norms) > norms_avg - alpha * norms_std))
        indices = norm_condition.nonzero()

        # Avoid pruning the whole layer
        if (len(indices) == layer.out_channels):
            c = random.choice(np.arange(0, layer.out_channels - 1, 1).tolist())
            indices = indices[1:]

        return indices.squeeze(dim=1).tolist()
            
    
    
    

