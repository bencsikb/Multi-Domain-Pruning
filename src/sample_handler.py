import pandas as pd
import os
from types import SimpleNamespace


class SampleHandler():
    def __init__(self, conf: SimpleNamespace) -> None:
        self.data_path = os.path.join(conf.samples.save_path, "data")
        self.label_path = os.path.join(conf.samples.save_path, "label")

        self.sample_container = {}
    
    def read_all_samples(self) -> None:

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
    
    def check_label_equality(self, ldf1, ldf2, decimals=3) -> bool:

        are_equal = True

        init_columns1 = [col for col in ldf1.columns if '_init' in col]
        init_columns2 = [col for col in ldf2.columns if '_init' in col]
        assert set(init_columns1) == set(init_columns2), "The DataFrames do not have the same _init columns."

        for col in init_columns1:
            if not ldf1[col].round(decimals).equals(ldf2[col].round(decimals)):
                print(f"Mismatch found in column: {col}")
                are_equal = False

        return are_equal

    
    @property
    def n_samples(self) -> int:
        return len(self.sample_container)