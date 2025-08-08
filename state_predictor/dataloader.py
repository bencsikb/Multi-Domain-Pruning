import os
import csv
import torch
import numpy as np
import pandas as pd
import torch
import pickle

from types import SimpleNamespace
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset



from state_predictor.coder import Coder


def create_pruning_dataloader(conf: SimpleNamespace, 
                              split_type: str, 
                              shuffle: bool = True, 
                              world_size: int = 1) -> DataLoader:

    if conf.model.type == "transformer":
        dataset = SPNDatasetTransformer(conf, split_type)
    else:
        dataset = SPNDataset(conf, split_type)
    
    batch_size = min(conf.train.batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])  
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=nw,
                            shuffle=shuffle,
                            collate_fn=dataset.collate_fn)
    return dataloader



class BaseSPNDataset(Dataset):
    def __init__(self,
                 conf: SimpleNamespace,
                 split_type: str,
                 data_folder: str = "data",
                 label_folder: str = "label"):
        self._conf = conf
        self._root_path = os.path.join(conf.data.root, split_type)
        self.data_path = os.path.join(self._root_path, data_folder)
        self.label_path = os.path.join(self._root_path, label_folder)
        self.cache_path = os.path.join(
            self._root_path, f"cache_{conf.data.cache_fantasy_name}.pkl")

        self.state_files = sorted(os.listdir(self.data_path))
        self.label_files = sorted(os.listdir(self.label_path))

        assert len(self.state_files) == len(self.label_files), (
            f"Mismatch: {len(self.state_files)} state vs {len(self.label_files)} label files"
        )

        self.coder = self._initialize_coder()

        if self._conf.data.is_read_from_cache:
            assert os.path.exists(self.cache_path), f"Missing cache file: {self.cache_path}"
            self._load_cache()
        else:
            self._build_cache()

    def _initialize_coder(self):
        state_df = pd.read_pickle(os.path.join(self.data_path, self.state_files[0]))
        label_df = pd.read_pickle(os.path.join(self.label_path, self.label_files[0]))
        alpha_range = self._conf.alpha.min_max_steps[:2]
        return Coder(state_df, label_df, alpha_range)

    def _parse_filename(self, filename):
        parts = filename.replace(".pkl", "").split('_')
        assert len(parts) == 3, f"Invalid filename format: {filename}"
        return f"{parts[0]}_{parts[1]}", int(parts[2])  # sequence_id, layer

    def _load_cache(self):
        with open(self.cache_path, "rb") as f:
            self.cache = pickle.load(f)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        return self.cache[index]

    def clear_cache(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        self.cache.clear()
    
    def _build_cache(self):
        raise NotImplementedError("Subclasses must implement `_build_cache()`.")

    def collate_fn(self):
        raise NotImplementedError("Subclasses must implement `collate_fn()`.")


class SPNDataset(BaseSPNDataset):

    def _build_cache(self):
        """Loads, processes, and stores dataset in a cache file."""
        self.cache = {}

        for index in range(len(self.state_files)):
            state_path = os.path.join(self.data_path, self.state_files[index])
            label_path = os.path.join(self.label_path, self.label_files[index])

            # Load state and label DataFrames
            state_df = pd.read_pickle(state_path)
            label_df = pd.read_pickle(label_path)

            # Filter for selected features
            state_df = state_df[self._conf.model.state_features]  

            # Encode state and label using the coder instance
            encoded_state = self.coder.encode_state(state_df, label_df)
            encoded_label = self.coder.encode_label(label_df)

            # Store in cache
            self.cache[index] = (encoded_state, encoded_label)

        # Save cache to file
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching."""
        data, label = zip(*batch)
        return torch.stack(data, 0), torch.stack(label, 0)



class SPNDatasetTransformer(BaseSPNDataset):
    def _build_cache(self):
        self.cache = {}
        sequence_dict = defaultdict(list)

        # Group files by sequence ID
        for sfile, lfile in zip(self.state_files, self.label_files):
            seq_id, layer = self._parse_filename(sfile)
            sequence_dict[seq_id].append((sfile, lfile, layer))

        # Sort files in each sequence chronologically
        for seq_id in sequence_dict:
            sequence_dict[seq_id].sort(key=lambda x: x[0])

        # Process each sequence as one training sample
        cache_idx = 0
        for (seq_id, samples) in sequence_dict.items():
            states, labels = [], []
            for sfile, lfile, layer_i in samples:

                if layer_i != 92:
                    continue
                
                state_df = pd.read_pickle(os.path.join(self.data_path, sfile))
                label_df = pd.read_pickle(os.path.join(self.label_path, lfile))

                state_df = state_df[self._conf.model.state_features]

                for i, state_row in state_df.iterrows():
                    
                    if i == 92:
                        label = label_df
                    else:
                        next_row = state_df.loc[i+1].to_frame().T.reset_index(drop=True)
                        label_n_params = next_row["prev_n_params"]
                        label_map = next_row["prev_map50"]
                        label_n_params_init = label_df["n_params_init"]
                        label_map_inint = label_df["map50_init"]
                        label = pd.DataFrame(     [(label_n_params.item(), label_n_params_init.item(), 
                                                    label_map.item(), label_map_inint.item())],
                                                    columns=["n_params", "n_params_init", "map50", "map50_init"]       
                                                )

                    state = self.coder.encode_state(state_row.to_frame().T.reset_index(drop=True), label_df)
                    label = self.coder.encode_label(label)

                    states.append(state)
                    labels.append(label)

                if len(states) >= 2:
                    inputs = torch.stack(states[:-1])   # (T-1, 3)
                    targets = torch.stack(labels[1:])   # (T-1, 2)
                    self.cache[cache_idx] = (inputs, targets)
                    cache_idx += 1
                else:
                    pass

        # Save the processed cache
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    @staticmethod
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        input_tensor = torch.stack(inputs, dim=0)   # (B, 93, 3)
        target_tensor = torch.stack(targets, dim=0) # (B, 93, 2)
        return input_tensor, target_tensor

    def _parse_filename(self, filename):
        """
        Returns:
        - sequence_id: 'a_b' → used for grouping
        - layer: int → the 'l' value (number of rows to keep from state_df)
        """
        parts = filename.replace(".pkl", "").split('_')  # remove extension if present
        assert len(parts) == 3, f"Filename {filename} does not follow a_b_l format"
        seq_id = int(parts[1])
        layer = int(parts[2])
        return seq_id, layer