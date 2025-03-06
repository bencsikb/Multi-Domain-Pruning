import pandas as pd
from typing import Tuple


class Coder:
    def __init__(self, 
                 sample_example: pd.DataFrame, 
                 label_example: pd.DataFrame,
                 alpha_range: Tuple,
                 encode_range: Tuple
                 ) -> None:

        
        self._encode_range = encode_range
        self._alpha_range = alpha_range
        self._set_param_ranges(sample_example, label_example)

        self._accuracy_range = [0, 1]
        self._params_range = [0, 64] # TODO

    def _set_param_ranges(self, sample: pd.DataFrame, label: pd.DataFrame):

        min_channels = sample[['in_ch', 'out_ch']].min().min()
        max_channels = sample[['in_ch', 'out_ch']].max().max()

        self._channel_range = (min_channels, max_channels) # Should this be defined for each layer individually?
        self._kernel_range = (sample['kernel'].min(), sample['kernel'].max())
        self._stride_range = (sample['stride'].min(), sample['stride'].max())
        self._pad_range = (sample['pad'].min(), sample['pad'].max())
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
        

    def encode_state(self, state):
        pass
    
    def encode_label(self, label):
        pass


    def decode_label(self, label):
        pass


# QUESTIONS:

# - mi jelezze a nem prunolást? Egy dedikált param vagy nem feltöltött state mátrix?
# - normálás minden layer channeljére külön-külön vagy a max channel méretre?
# - milyen normálási range legyen vagy milyen aktiváció, hogy az 1-nél nagyobb prop (dmap) is működjön?