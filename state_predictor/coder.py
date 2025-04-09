import torch
import numpy as np
import pandas as pd
from pandas.core.series import Series
from typing import Tuple

from state_predictor.utils import normalize


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

    
    def _calculate_dmap(self, map: Series, map_init: Series) -> float:
        """
        Parameters:
            map, map_init (Series): A one-element Series representing the current mAP and the initial mAP.
        """
        return 1 - (map.item() / map_init.item())

    def _calculate_spars(self, n_params: Series, n_params_init: Series) -> float:
        """
        Parameters:
            n_params, n_params_init (Series): A one-element Series representing the current and the 
                                              initial nuber of parameters in the model.
        """
        return 1 - (n_params.item() / n_params_init.item())

    def _determine_pruned_area(self):
        """
        Not sure if needed, but get the pruned param from the data df and 
        leave the rest of the data for the layer if 1, delete if 0.
        """
        pass
        
    def encode_state(self, state: pd.DataFrame) -> torch.Tensor:
        # shape [n_features+1, n_prunable_layers]
        n_features = len(state.columns)
        encoded_state = np.full((self.n_prunable_layers, n_features), -1.0, dtype=np.float32)

        col_range_map = {
            'alpha': (0, self._alpha_range),
            'is_pruned': (1, self._is_pruned_range),
            'in_ch': (2, self._channel_range),
            'out_ch': (3, self._channel_range),
            'kernel': (4, self._kernel_range),
            'stride': (5, self._stride_range),
            'pad': (6, self._pad_range),
            'n_pruned_ch': (7, self._channel_range),
        }

        for col, (idx, range_) in col_range_map.items():
            if col in state.columns:
                encoded_state[:, idx] = normalize(state[col].values, range_)

        # Make it one-dimensional
        encoded_state = encoded_state.flatten()

        return torch.tensor(encoded_state, dtype=torch.float32)


    
    def encode_label(self, label: pd.DataFrame) -> torch.Tensor:
        # [sparsity, dmap]
        encoded_label = torch.zeros([2])  

        sparsity = self._calculate_spars(label['n_params'], label['n_params_init'])
        dmap = self._calculate_dmap(label['map50'], label['map50_init'])
        encoded_label[0] = normalize(sparsity, value_range=(0, 1))
        encoded_label[1] = normalize(dmap, value_range=(0, 1))
        
        return torch.Tensor(encoded_label)


    def decode_label(self, label):
        pass


# QUESTIONS:

# - mi jelezze a nem prunolást? Egy dedikált param vagy nem feltöltött state mátrix?
# - normálás minden layer channeljére külön-külön vagy a max channel méretre?
# - milyen normálási range legyen vagy milyen aktiváció, hogy az 1-nél nagyobb prop (dmap) is működjön?