import pandas as pd
import os
import numpy as np
from types import SimpleNamespace


class SampleHandler():
    def __init__(self, conf: SimpleNamespace) -> None:
        self.data_path = os.path.join(conf.samples.save_path, "data")
        self.label_path = os.path.join(conf.samples.save_path, "label")

        self.sample_container = {}
    
    def read_all_samples(self) -> None:
        # TODO: handle exception whem only data or only label file exists

        for filename in os.listdir(self.data_path):
            if filename.endswith('.pkl'):
                data_df = pd.read_pickle(os.path.join(self.data_path, filename))
                label_df = pd.read_pickle(os.path.join(self.label_path, filename))
                self.add_sample(data_df, label_df)               

    def _df_to_string(self, df) -> str:
        # Convert all values to a single string by flattening and concatenating
        return ''.join(map(str, df.values.flatten()))
    
    def add_sample(self, data_df, label_df) -> None:
        sample_string = self._df_to_string(data_df)
        if sample_string not in self.sample_container:
            self.sample_container[sample_string] = (data_df, label_df)
        else:
            print(f"Sample already exists!")
   

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