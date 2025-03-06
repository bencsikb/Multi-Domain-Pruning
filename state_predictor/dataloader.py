import os
import csv
import torch
import numpy as np
import pandas as pd
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset

from state_predictor.coder import Coder


def create_pruning_dataloader(data_path, sample_path, cache_path, cache_ext, batch_size, shuffle, world_size=1):
    dataset = SPNDataset(data_path, sample_path, cache_path, cache_ext)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=nw,
                            shuffle=shuffle,
                            collate_fn=SPNDataset.collate_fn)
    return dataloader, dataset


class SPNDataset(Dataset):
    def __init__(self, 
                 root_path: str, 
                 conf: SimpleNamespace, 
                 data_folder: str = "data", 
                 label_folder: str = "label"):
        
        self.root_path = root_path
        self.conf = conf
        self.data_path = os.path.join(root_path, data_folder)
        self.label_path = os.path.join(root_path, label_folder)

        self.state_files = os.listdir(self.data_path)
        self.label_files = os.listdir(self.label_path)

        assert len(self.state_files) == len(self.label_files), (
            f"Mismatch in file counts: {len(self.state_files)} state files vs {len(self.label_files)} label files."
        )      

        self.coder = self._initialize_coder() # transform
    
    def _initialize_coder(self) -> Coder:

        state_example_df = pd.read_pickle(self.state_files[0])
        label_example_df = pd.read_pickle(self.label_files[0])
        alpha_range = self.conf.alpa.min_max_steps[:2]
        coder = Coder(state_example_df, label_example_df, alpha_range)
        return coder

    def __len__(self):
        return len(self.state_files)


    def __getitem__(self, index):

        # Load state and label df
        state_df = pd.read_pickle(self.state_files[index])
        label_df = pd.read_pickle(self.label_files[index])

        # Encode state and label df
        encoded_state = Coder.encode_state(self.states[index])
        encoded_label = Coder.encode_label(self.labels[index])
        
        return encoded_state, encoded_label

       
    def collate_fn(batch):
        data, label = zip(*batch)
        return torch.stack(data, 0), torch.cat(label, 0)