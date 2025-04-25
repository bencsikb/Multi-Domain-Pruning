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
        """
        Encode pruning state DataFrame into a 1D tensor.
        
        Returns:
            torch.Tensor of shape [n_prunable_layers * n_active_features]
        """
        col_range_map = {
            'alpha': self._alpha_range,
            'is_pruned': self._is_pruned_range,
            'in_ch': self._channel_range,
            'out_ch': self._channel_range,
            'kernel': self._kernel_range,
            'stride': self._stride_range,
            'pad': self._pad_range,
            'n_pruned_ch': self._channel_range,
        }

        encoded_state = []

        for col, range_ in col_range_map.items():
            if col in state.columns:
                normalized = normalize(state[col].values, range_).astype(np.float32)  # shape [n_prunable_layers]
                encoded_state.append(normalized)  

        if not encoded_state:
            raise ValueError("No recognized columns found in the input state DataFrame.")

        # Stack to shape [n_features_used, n_prunable_layers] → transpose to [n_prunable_layers, n_features_used]
        encoded_matrix = np.stack(encoded_state, axis=0).T

        return torch.tensor(encoded_matrix.flatten(), dtype=torch.float32)


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