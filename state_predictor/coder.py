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

        self._spars_range = (0, 1)
        self._dmap_range = (0, 1)
        
        # self._map_range = (0, label['map50_init'])
        # self._params_range = (0, label['n_params_init'])

    
    def _calculate_dmap(self, map: pd.Series, map_init: pd.Series) -> pd.Series:
        """
        Calculates delta mAP.

        Parameters:
            map (Series): Series of current mAP values.
            map_init (Series): One-element Series with the initial mAP.

        Returns:
            Series: 1 - (map / init_map), elementwise.
        """
        init_val = map_init.item()  # scalar
        return 1 - (map / init_val)


    def _calculate_spars(self, n_params: pd.Series, n_params_init: pd.Series) -> pd.Series:
        """
        Calculates sparsity.

        Parameters:
            n_params (Series): Series of current number of parameters.
            n_params_init (Series): One-element Series with the initial number of parameters.

        Returns:
            Series: 1 - (n_params / init_n_params), elementwise.
        """
        init_val = n_params_init.item()  # scalar
        return 1 - (n_params / init_val)

    def _determine_pruned_area(self):
        """
        Not sure if needed, but get the pruned param from the data df and 
        leave the rest of the data for the layer if 1, delete if 0.
        """
        pass
        
    def encode_state(self, state: pd.DataFrame, label: pd.DataFrame) -> torch.Tensor:
        """
        Encode a model pruning state into a 1D tensor.

        Parameters:
            state (pd.DataFrame): Pruning state of the model.
            label (pd.DataFrame): Contains scalar values like 'n_params_init' and 'map50_init'.
            do_normalize (bool): Whether to normalize the features.

        Returns:
            torch.Tensor: Flattened feature tensor of shape [n_features * n_prunable_layers]
        """
        # We'll track active features ourselves
        encoded_state = []

        # Define mappings
        col_range_map = {
            'alpha': self._alpha_range,
            'is_pruned': self._is_pruned_range,
            'in_ch': self._channel_range,
            'out_ch': self._channel_range,
            'kernel': self._kernel_range,
            'stride': self._stride_range,
            'pad': self._pad_range,
            'n_pruned_ch': self._channel_range,
            'prev_n_params': self._spars_range,
            'prev_map50': self._dmap_range,
        }

        for col in col_range_map:
            if col not in state.columns:
                continue

            if col == "prev_n_params":
                init_val = label['n_params_init'].item() if isinstance(label['n_params_init'], pd.Series) else label['n_params_init']
                values = self._calculate_spars(state[col], pd.Series([init_val]))
            elif col == "prev_map50":
                init_val = label['map50_init'].item() if isinstance(label['map50_init'], pd.Series) else label['map50_init']
                values = self._calculate_dmap(state[col], pd.Series([init_val]))
            else:
                values = state[col]

            values = normalize(values.values, col_range_map[col])
            encoded_state.append(values.astype(np.float32))

        # Stack: shape [n_features_active, n_prunable_layers] → transpose
        encoded_state = np.stack(encoded_state, axis=0).T  # shape [n_prunable_layers, n_features_active]

        # Flatten
        return torch.tensor(encoded_state.flatten(), dtype=torch.float32)

    
    def encode_label(self, label: pd.DataFrame) -> torch.Tensor:
        # [sparsity, dmap]
        encoded_label = torch.zeros([2])  

        sparsity = self._calculate_spars(label['n_params'], label['n_params_init'])
        dmap = self._calculate_dmap(label['map50'], label['map50_init'])
        encoded_label[0] = normalize(sparsity.item(), value_range=(0, 1))
        encoded_label[1] = normalize(dmap.item(), value_range=(0, 1))

        
        return torch.Tensor(encoded_label)


    def decode_label(self, label):
        pass
