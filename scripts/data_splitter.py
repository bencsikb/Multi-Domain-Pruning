import os
import shutil
import pandas as pd

def copy_dataset_splits(csv_path, orig_image_path, orig_label_path, output_base_path):
    # Load split data
    df = pd.read_csv(csv_path)

    # Define output paths for images and labels
    split_folders = {
        "train": os.path.join(output_base_path, "training"),
        "val": os.path.join(output_base_path, "validation"),
        "test": os.path.join(output_base_path, "testing"),
    }

    for split, base_path in split_folders.items():
        os.makedirs(os.path.join(base_path, "images_newsplit"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "labels_newsplit"), exist_ok=True)

    # Copy files to respective folders
    for _, row in df.iterrows():
        filename = row["filename"]
        split = row["split"]

        # Define source and destination paths
        src_image = os.path.join(orig_image_path, filename)
        src_label = os.path.join(orig_label_path, filename.replace(".png", ".txt"))  # Assuming label files are .txt
        
        dst_image = os.path.join(split_folders[split], "images_newsplit", filename)
        dst_label = os.path.join(split_folders[split], "labels_newsplit", filename.replace(".png", ".txt"))

        # Copy image
        if os.path.exists(src_image):
            shutil.copy2(src_image, dst_image)

        # Copy label (if it exists)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    print("Dataset successfully copied into new splits!")

# Example usage
csv_path = "dataset_splits.csv"  # Path to split CSV
orig_image_path = "/data2/blanka/DATASETS/KITTI/original/image_2"  # Replace with actual path
orig_label_path = "/data2/blanka/DATASETS/KITTI/original/label_2_yolo"  # Replace with actual path
output_base_path = "/data2/blanka/DATASETS/KITTI"  # Replace with actual path

copy_dataset_splits(csv_path, orig_image_path, orig_label_path, output_base_path)
