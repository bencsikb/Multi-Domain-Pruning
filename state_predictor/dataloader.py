import os
import csv
import torch
import numpy as np

from torch.utils.data import DataLoader


def create_pruning_dataloader(data_path, sample_path, cache_path, cache_ext, batch_size, shuffle, world_size=1):
    dataset = LoadPruningData(data_path, sample_path, cache_path, cache_ext)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=shuffle,
                                             collate_fn=LoadPruningData.collate_fn)
    return dataloader, dataset


class LoadPruningData():
    def __init__(self, data_path):
        pass
        
        self.data_path = data_path
        self.label_path = ""

        data_df, label_df = self._load_data_and_label_dfs()
        encoded_data, encoded_label = encoder.encode(data_df, label_df)
        

    def _load_data_and_label_dfs(self):
        pass
    
    def __len__(self):
        return self.state_data.shape[0]


    def __getitem__(self, index):
        
        return self.augment_state(self.state_data[index, ...]), self.label_data[index, ...]

       
    def collate_fn(batch):
        data, label = zip(*batch)
        return torch.stack(data, 0), torch.cat(label, 0)