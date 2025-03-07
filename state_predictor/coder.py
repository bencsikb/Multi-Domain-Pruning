import torch
import numpy as np
import pandas as pd
from typing import Tuple

from utils import normalize, denormalize


class Coder:
    def __init__(self, 
                 state_example: pd.DataFrame, 
                 label_example: pd.DataFrame,
                 alpha_range: Tuple[float],
                 encode_range: Tuple[float] = (-1, 1)
                 ) -> None:

        
        self._encode_range = encode_range
        self._alpha_range = alpha_range
        self._set_param_ranges(state_example, label_example)

        self.n_prunable_layers = state_example.shape[0]

    def _set_param_ranges(self, state: pd.DataFrame, label: pd.DataFrame):

        min_channels = state[['in_ch', 'out_ch']].min().min()
        max_channels = state[['in_ch', 'out_ch']].max().max()

        self._channel_range = (min_channels, max_channels) # Should this be defined for each layer individually?
        self._kernel_range = (state['kernel'].min(), state['kernel'].max())
        self._stride_range = (state['stride'].min(), state['stride'].max())
        self._pad_range = (state['pad'].min(), state['pad'].max())
        self._is_pruned_range = (0, 1)
        
        self._map_range = (0, label['map50_init'])
        self._params_range = (0, label['n_params_init'])

    
    def _calculate_dmap(self):
        pass
        # 

    def _calculate_spars(self):
        pass

    def _determine_pruned_area(self):
        """
        Not sure if needed, but get the pruned param from the data df and 
        leave the rest of the data for the layer if 1, delete if 0.
        """
        pass
        
    def encode_state(self, state: pd.DataFrame) -> torch.Tensor:
        # shape [n_features+1, n_prunable_layers]
        n_features = 8

        encoded_state = np.full((self.n_prunable_layers, n_features), -1.0, dtype=np.float32)

        encoded_state[:, 0] = normalize(state['alpha'].values, self._alpha_range)
        encoded_state[:, 1] = normalize(state['is_pruned'].values, self._is_pruned_range)
        encoded_state[:, 2] = normalize(state['in_ch'].values, self._channel_range)
        encoded_state[:, 3] = normalize(state['out_ch'].values, self._channel_range)
        encoded_state[:, 4] = normalize(state['kernel'].values, self._kernel_range)
        encoded_state[:, 5] = normalize(state['stride'].values, self._stride_range)
        encoded_state[:, 6] = normalize(state['pad'].values, self._pad_range)
        encoded_state[:, 7] = normalize(state['n_pruned_ch'].values, self._channel_range)

        return torch.tensor(encoded_state, dtype=torch.float32)


    
    def encode_label(self, label: pd.DataFrame) -> torch.Tensor:
        pass
        # [sparsity, dmap]
        encoded_label = torch.zeros([2])  

        sparsity = 1 - (label['n_params'] / label['n_params_init'])
        dmap = 1 - (label['map50' / label['map50_init']])
        encoded_label[0] = normalize(sparsity, value_range=(0, 1))
        encoded_label[1] = normalize(dmap, value_range=(0, 1))

        return torch.Tensor(encoded_label)


    def decode_label(self, label):
        pass


# QUESTIONS:

# - mi jelezze a nem prunolást? Egy dedikált param vagy nem feltöltött state mátrix?
# - normálás minden layer channeljére külön-külön vagy a max channel méretre?
# - milyen normálási range legyen vagy milyen aktiváció, hogy az 1-nél nagyobb prop (dmap) is működjön?