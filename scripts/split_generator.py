import os
import random
import pandas as pd
import re

def make_divisible(value, batch_size):
    return value - (value % batch_size)

def extract_number(filename):
    """Extracts the first number found in a filename for numeric sorting."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')

def split_dataset(image_folder, output_csv="dataset_splits.csv", ratios=[0.8, 0.1, 0.1], batch_size=16, ext=".png", seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    assert len(ratios) == 3, "Ratios must have exactly three values: [train, val, test]"

    train_ratio, val_ratio, test_ratio = ratios

    # Get and sort .png files numerically
    all_images = [f for f in os.listdir(image_folder) if f.endswith(ext)]
    all_images.sort(key=extract_number)

    # Shuffle after numeric sort for randomness
    random.seed(seed)
    random.shuffle(all_images)

    total_images = len(all_images)

    # Compute and adjust split sizes
    train_count = make_divisible(int(total_images * train_ratio), batch_size)
    val_count = make_divisible(int(total_images * val_ratio), batch_size)
    test_count = make_divisible(total_images - train_count - val_count, batch_size)

    remaining = total_images - (train_count + val_count + test_count)

    if remaining >= batch_size:
        additions = make_divisible(remaining, batch_size) // batch_size
        if additions >= 2:
            train_count += batch_size
            val_count += batch_size
            test_count += (remaining - 2 * batch_size)
        elif additions == 1:
            train_count += batch_size

    # Split dataset
    train_files = all_images[:train_count]
    val_files = all_images[train_count:train_count + val_count]
    test_files = all_images[train_count + val_count:train_count + val_count + test_count]

    # Create DataFrame and sort it numerically
    df = pd.DataFrame({
        "filename": train_files + val_files + test_files,
        "split": ["train"] * len(train_files) + 
                 ["val"] * len(val_files) + 
                 ["test"] * len(test_files)
    })

    df = df.sort_values(by="filename", key=lambda col: col.map(extract_number)).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset split saved to {output_csv}")

    # Print Summary
    print("\n=== Dataset Split Summary ===")
    print(f"Total images: {total_images}")
    print(f"Train set: {train_count} images (Divisible by {batch_size} ✅)")
    print(f"Validation set: {val_count} images (Divisible by {batch_size} ✅)")
    print(f"Test set: {test_count} images (Divisible by {batch_size} ✅)")
    print(f"Remaining images after adjustment: {total_images - (train_count + val_count + test_count)}")

if __name__ == "__main__":

    image_folder = "/data/blanka/DATASETS/SPN/YOLOv8x_newsplit/original/data"  
    split_dataset(image_folder, output_csv="dataset_splits.csv", ratios=[0.5,0.5,0.0], batch_size=1, ext=".pkl")
