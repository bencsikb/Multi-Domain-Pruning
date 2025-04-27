import pandas as pd
import os
import numpy as np
from types import SimpleNamespace
import logging


class SampleHandler():
    def __init__(self, conf: SimpleNamespace) -> None:
        self.data_path = os.path.join(conf.samples.save_path, "data")
        self.label_path = os.path.join(conf.samples.save_path, "label")

        self.sample_container = {}
        self._init_counters()
    
    def read_all_samples(self) -> None:
        # TODO: handle exception whem only data or only label file exists

        for filename in os.listdir(self.data_path):
            if filename.endswith('.pkl'):
                data_df = pd.read_pickle(os.path.join(self.data_path, filename))
                label_df = pd.read_pickle(os.path.join(self.label_path, filename))
                self.add_sample(data_df, label_df)   

    def _init_counters(self):

        filenames = [f for f in os.listdir(self.data_path) if f.endswith('.pkl')]

        if not filenames: 
            self._sample_counter = 0
            self._model_counter = 0
            self._layer_counter = 0
            return

        parsed = []
        for filename in filenames:
            parts = filename.split("_")
            first_part = int(parts[0])
            parsed.append((first_part, filename))

        # Sort by the first number
        parsed.sort(key=lambda x: x[0])

        # Group by first_part and take the last for each group
        from collections import defaultdict

        grouped = defaultdict(list)
        for first_part, filename in parsed:
            grouped[first_part].append(filename)

        # Take the last entry for the highest first_part
        last_first_part = max(grouped.keys())
        last_filename = grouped[last_first_part][-1]

        # Extract sample, model, layer counters from filename
        parts = last_filename.replace('.pkl', '').split('_')
        self._sample_counter = int(parts[0])
        self._model_counter = int(parts[1])
        self._layer_counter = int(parts[2])


    def _df_to_string(self, df) -> str:
        # Convert all values to a single string by flattening and concatenating
        return ''.join(map(str, df.values.flatten()))
    
    def add_sample(self, data_df, label_df) -> None:
        sample_string = self._df_to_string(data_df)
        if sample_string not in self.sample_container:
            self.sample_container[sample_string] = (data_df, label_df)
        else:
            logging.warning("Sample already exists in the sample_handler!")
   

    def is_existing_sample(self, data_df) -> bool:
        sample_string = self._df_to_string(data_df)
        return sample_string in self.sample_container

    def retrieve_sample(self, data_df):
        sample_string = self._df_to_string(data_df)
        data_df, label_df = self.sample_container.get(sample_string, None)
        return label_df
    
    def check_label_equality(self, saved_df, init_df, decimals=3) -> bool:

        saved_init_df = saved_df.filter(like='_init', axis=1)

        l1 = saved_init_df.to_numpy().flatten().tolist()
        l2 = init_df.to_numpy().flatten().tolist()

        assert len(l1) == len(l2), "The lists are of different lengths."

        l1 = np.round(l1, decimals)
        l2 = np.round(l2, decimals)

        return np.array_equal(l1, l2)

    
    @property
    def n_samples(self) -> int:
        return len(self.sample_container)
    
    @property
    def model_counter(self) -> int:
        return self._model_counter
    
    @property
    def layer_counter(self) -> int:
        return self._layer_counter
    
    def increment_model_counter_if_needed(self, layer_index: int) -> None:
        if layer_index == 0:
            self._model_counter += 1

    def set_layer_counter(self, layer_index: int) -> None:
        self._layer_counter = layer_index