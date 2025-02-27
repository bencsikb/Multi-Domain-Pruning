import os
import random
import pandas as pd

def make_divisible(value, batch_size):
    """Rounds down the value to the nearest multiple of batch_size."""
    return value - (value % batch_size)

def split_dataset(image_folder, output_csv="dataset_splits.csv", ratios=[0.8, 0.1, 0.1], batch_size=16, seed=42):
    # Ensure ratios sum to 1
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
    assert len(ratios) == 3, "Ratios must have exactly three values: [train, val, test]"

    train_ratio, val_ratio, test_ratio = ratios

    # Get all .png files in the folder
    all_images = [f for f in os.listdir(image_folder) if f.endswith(".png")]

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(all_images)

    total_images = len(all_images)

    # Compute initial split sizes
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - (train_count + val_count)  # Ensure all images are included

    # Make all sets divisible by batch_size
    train_count = make_divisible(train_count, batch_size)
    val_count = make_divisible(val_count, batch_size)
    test_count = make_divisible(test_count, batch_size)

    # Adjust counts to maintain total size
    remaining = total_images - (train_count + val_count + test_count)

    # Distribute remaining images in multiples of batch_size
    if remaining >= batch_size:
        additions = make_divisible(remaining, batch_size) // batch_size
        if additions >= 2:
            train_count += batch_size
            val_count += batch_size
            test_count += (remaining - 2 * batch_size)  # Ensuring all are used
        elif additions == 1:
            train_count += batch_size  # Assigning extra images to train set

    # Split dataset
    train_files = all_images[:train_count]
    val_files = all_images[train_count:train_count + val_count]
    test_files = all_images[train_count + val_count:train_count + val_count + test_count]

    # Create a DataFrame
    df = pd.DataFrame({"filename": train_files + val_files + test_files,
                       "split": ["train"] * len(train_files) + 
                                ["val"] * len(val_files) + 
                                ["test"] * len(test_files)})

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

# Example usage
image_folder = "/data2/blanka/DATASETS/KITTI/original/image_2"  # Replace with your actual folder path
split_dataset(image_folder, output_csv="dataset_splits.csv", ratios=[0.7,0.15,0.15], batch_size=8)
