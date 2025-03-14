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


import os
import torch
import pandas as pd
import pickle
from types import SimpleNamespace
from torch.utils.data import Dataset

class SPNDataset(Dataset):
    def __init__(self, 
                 root_path: str, 
                 conf: SimpleNamespace, 
                 data_folder: str = "data", 
                 label_folder: str = "label",
                 cache_file: str = "cache.pkl",
                 rebuild_cache: bool = False):
        
        self.root_path = root_path
        self.conf = conf
        self.data_path = os.path.join(root_path, data_folder)
        self.label_path = os.path.join(root_path, label_folder)
        self.cache_path = os.path.join(root_path, cache_file)

        # Ensure lists are sorted to maintain correspondence
        self.state_files = sorted(os.listdir(self.data_path))
        self.label_files = sorted(os.listdir(self.label_path))

        assert len(self.state_files) == len(self.label_files), (
            f"Mismatch in file counts: {len(self.state_files)} state files vs {len(self.label_files)} label files."
        )      

        self.coder = self._initialize_coder()

        # Load cache if available, else create it
        if os.path.exists(self.cache_path) and not rebuild_cache:
            self._load_cache()
        else:
            self._build_cache()

    def _initialize_coder(self):
        """Initialize the coder with an example file."""
        state_example_df = pd.read_pickle(os.path.join(self.data_path, self.state_files[0]))
        label_example_df = pd.read_pickle(os.path.join(self.label_path, self.label_files[0]))
        alpha_range = self.conf.alpa.min_max_steps[:2]
        return Coder(state_example_df, label_example_df, alpha_range)

    def _build_cache(self):
        """Loads, processes, and stores dataset in a cache file."""
        self.cache = {}

        for index in range(len(self.state_files)):
            state_path = os.path.join(self.data_path, self.state_files[index])
            label_path = os.path.join(self.label_path, self.label_files[index])

            # Load state and label DataFrames
            state_df = pd.read_pickle(state_path)
            label_df = pd.read_pickle(label_path)

            # Encode state and label using the coder instance
            encoded_state = self.coder.encode_state(state_df)
            encoded_label = self.coder.encode_label(label_df)

            # Store in cache
            self.cache[index] = (encoded_state, encoded_label)

        # Save cache to file
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def _load_cache(self):
        """Loads dataset from an existing cache file."""
        with open(self.cache_path, "rb") as f:
            self.cache = pickle.load(f)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        """Retrieve preprocessed data from cache."""
        return self.cache[index]

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching."""
        data, label = zip(*batch)
        return torch.stack(data, 0), torch.cat(label, 0)

    def clear_cache(self):
        """Deletes the cache file and clears in-memory cache."""
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        self.cache.clear()
