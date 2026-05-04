import os
import shutil
import random
import urllib.parse
from pathlib import Path


def create_dataset():
    base_dir = Path(
        r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies"
    )
    label_dir = base_dir / "proposed_bboxes" / "v4" / "labels"
    image_dir = base_dir / "clahe_leaves"
    output_dir = base_dir / "preannotations" / "data_for_preannotation_model_v4"

    # Create directory structure
    for split in ["train", "validation"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Get all label files
    if not label_dir.exists():
        print(f"Error: Label directory not found at {label_dir}")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    random.seed(42)  # Set seed for reproducible splits
    random.shuffle(label_files)

    # Split dataset (10% validation)
    val_size = int(len(label_files) * 0.1)
    splits = {"validation": label_files[:val_size], "train": label_files[val_size:]}

    processed_count = 0
    missing_images = 0

    for split_name, files in splits.items():
        print(f"Processing {split_name} split ({len(files)} files)...")
        for label_file in files:
            # Parse the original image name from the encoded label filename
            # Example: 0252fc1c__whiteflies%5Cclahe_leaves%5Cimage-1197.txt
            decoded_name = urllib.parse.unquote(label_file)

            # Extract the actual filename (e.g., image-1197.txt)
            # It might have backslashes or forward slashes from the original path
            clean_basename = decoded_name.replace("\\", "/").split("/")[-1]

            # The label file is .txt, we expect the image to be .jpg
            image_name = clean_basename.replace(".txt", ".jpg")
            clean_label_name = clean_basename

            image_path = image_dir / image_name
            label_path = label_dir / label_file

            if not image_path.exists():
                print(
                    f"Warning: Image not found for label {label_file} (expected {image_name})"
                )
                missing_images += 1
                continue

            # Copy image to the new structure
            shutil.copy2(image_path, output_dir / split_name / "images" / image_name)

            # Copy and rename label to match image name exactly (YOLO format requirement)
            shutil.copy2(
                label_path, output_dir / split_name / "labels" / clean_label_name
            )

            processed_count += 1

    print("\nDataset creation complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Missing images: {missing_images}")
    print(f"Train size: {len(splits['train'])}")
    print(f"Validation size: {len(splits['validation'])}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    create_dataset()
