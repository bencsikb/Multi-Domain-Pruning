import pickle
import os
import pandas as pd
from IPython.display import display

def numeric_sort(file_list):
    return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0]))

src_data_path = "/data/blanka/DATASETS/SPN/YOLOv8x_newsplit/original_copy/data"
src_label_path = "/data/blanka/DATASETS/SPN/YOLOv8x_newsplit/original_copy/label"

dst_data_path = "/data/blanka/DATASETS/SPN/YOLOv8x_newsplit/original_extended/data"
dst_label_path = "/data/blanka/DATASETS/SPN/YOLOv8x_newsplit/original_extended/label"

samples = numeric_sort(os.listdir(src_data_path))
previous_dfs = []

for sample in samples:
    state_df = pd.read_pickle(os.path.join(src_data_path, sample))
    label_df = pd.read_pickle(os.path.join(src_label_path, sample))

    last_pruned_index = state_df[state_df["is_pruned"] == 0].index[0]

    # Initialize columns
    state_df["prev_n_params"] = 0.0
    state_df["prev_map50"] = 0.0

    match_found = False

    for prev_sample_name, prev_df in reversed(previous_dfs):  # <--- Reversed here
        prev_clean = prev_df.drop(columns=["prev_n_params", "prev_map50", "in_ch", "out_ch", "n_pruned_ch"], errors="ignore")
        state_clean = state_df.drop(columns=["prev_n_params", "prev_map50", "in_ch", "out_ch", "n_pruned_ch"], errors="ignore")

        state_clean.loc[state_clean.index[last_pruned_index - 1], "is_pruned"] = 0

        if prev_clean.shape[0] >= last_pruned_index:
            if state_clean.iloc[:last_pruned_index].equals(prev_clean.iloc[:last_pruned_index]):
                # Reuse from matching sample
                state_df.loc[:last_pruned_index - 1, "prev_n_params"] = prev_df.loc[:last_pruned_index - 1, "prev_n_params"]
                state_df.loc[:last_pruned_index - 1, "prev_map50"] = prev_df.loc[:last_pruned_index - 1, "prev_map50"]
                print(f"Match found for {sample}, using data from {prev_sample_name}")
                match_found = True
                break  # still break after match

    # Insert current label values at cutoff point
    state_df.loc[last_pruned_index, "prev_n_params"] = label_df["n_params"].values[0]
    state_df.loc[last_pruned_index, "prev_map50"] = label_df["map50"].values[0]

    # Add to previous samples
    previous_dfs.append((sample, state_df.copy()))

    # Roll
    rolled_df = state_df.copy()
    rolled_df.loc[rolled_df.index[2:], "prev_n_params"] = state_df["prev_n_params"].iloc[1:-1].values
    rolled_df.loc[rolled_df.index[2:], "prev_map50"] = state_df["prev_map50"].iloc[1:-1].values

    if last_pruned_index + 1 < len(rolled_df):
        rolled_df.loc[rolled_df.index[last_pruned_index + 1], "prev_n_params"] = 0.0
        rolled_df.loc[rolled_df.index[last_pruned_index + 1], "prev_map50"] = 0.0

    rolled_df.to_pickle(os.path.join(dst_data_path, sample))
