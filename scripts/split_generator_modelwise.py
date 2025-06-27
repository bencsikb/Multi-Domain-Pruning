import os
import random
import pandas as pd
import re

def extract_modelwise_number(filename):
    # Extracts the modelwise number from names like "789_12_85"
    matches = re.findall(r'\d+', filename)
    return int(matches[1]) if len(matches) >= 2 else None

def split_by_modelwise_group(image_folder, output_csv="dataset_splits.csv",
                             ratios=[0.8, 0.1, 0.1], ext=".png", seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    assert len(ratios) == 3, "Provide [train, val, test] ratios"
    
    all_files = [f for f in os.listdir(image_folder) if f.endswith(ext)]
    group_dict = {}
    for f in all_files:
        group_id = extract_modelwise_number(f)
        if group_id is not None:
            group_dict.setdefault(group_id, []).append(f)

    group_ids = sorted(group_dict.keys())
    random.seed(seed)
    random.shuffle(group_ids)

    total_groups = len(group_ids)
    train_count = int(total_groups * ratios[0])
    val_count = int(total_groups * ratios[1])
    test_count = total_groups - train_count - val_count

    train_ids = group_ids[:train_count]
    val_ids = group_ids[train_count:train_count + val_count]
    test_ids = group_ids[train_count + val_count:]

    split_map = {}
    for gid in train_ids:
        for fname in group_dict[gid]:
            split_map[fname] = "train"
    for gid in val_ids:
        for fname in group_dict[gid]:
            split_map[fname] = "val"
    for gid in test_ids:
        for fname in group_dict[gid]:
            split_map[fname] = "test"

    df = pd.DataFrame({
        "filename": list(split_map.keys()),
        "split": [split_map[f] for f in split_map]
    })
    df["modelwise_number"] = df["filename"].map(extract_modelwise_number)
    df = df.sort_values(by=["modelwise_number", "filename"]).reset_index(drop=True)
    df.drop("modelwise_number", axis=1, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset split saved to {output_csv}")
    print(f"Unique modelwise groups: {total_groups}")
    print(f"Train: {len(train_ids)} groups, Val: {len(val_ids)} groups, Test: {len(test_ids)} groups")

if __name__ == "__main__":
    image_folder = "/data/blanka/DATASETS/SPN/YOLOv8x/data_shifted"
    split_by_modelwise_group(image_folder, output_csv="SPN_modelwise_7030.csv",
                             ratios=[0.7, 0.3, 0.0], ext=".pkl", seed=42)
