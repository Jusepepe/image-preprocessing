import os
import json
import uuid
import cv2

def main():
    json_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\label_studio_predictions_v5.json"
    new_data_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\labels\new_data"
    images_dir = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\clahe_leaves"
    
    # Prefix for Label Studio Local Storage
    IMAGE_URL_PREFIX = "/data/local-files/?d=whiteflies%5Cclahe_leaves%5C"

    print(f"Reading JSON from {json_path}")
    with open(json_path, "r") as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks.")

    # Get new data base names
    new_data_files = [f for f in os.listdir(new_data_path) if f.endswith(".txt")]
    new_base_names = [os.path.splitext(f)[0] for f in new_data_files]
    new_base_names_set = set(new_base_names)

    # Filter out tasks that are in new_data
    filtered_tasks = []
    removed_count = 0
    for task in tasks:
        img_url = task["data"]["image"]
        
        # Extract base name from url
        # e.g., /data/local-files/?d=whiteflies%5Cclahe_leaves%5Cimage-877.jpg
        # get part after last %5C or /
        filename = img_url.split("%5C")[-1].split("/")[-1]
        base_name = os.path.splitext(filename)[0]
        
        if base_name in new_base_names_set:
            removed_count += 1
        else:
            filtered_tasks.append(task)

    print(f"Removed {removed_count} existing tasks that are found in new_data.")

    # Create new tasks
    added_count = 0
    for base_name in new_base_names:
        txt_path = os.path.join(new_data_path, f"{base_name}.txt")
        
        # Find corresponding image
        img_path = None
        img_filename = None
        for ext in [".jpg", ".png", ".jpeg"]:
            temp_path = os.path.join(images_dir, f"{base_name}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                img_filename = f"{base_name}{ext}"
                break
        
        if img_path is None:
            print(f"Warning: Image for {base_name} not found in {images_dir}. Skipping.")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
            
        orig_h, orig_w = img.shape[:2]

        task = {
            "data": {"image": f"{IMAGE_URL_PREFIX}{img_filename}"},
            "predictions": [{"result": []}],
        }
        
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert YOLO normalized coordinates to percentages for Label Studio
                x_min = x_center - (width / 2.0)
                y_min = y_center - (height / 2.0)
                
                x_perc = x_min * 100.0
                y_perc = y_min * 100.0
                w_perc = width * 100.0
                h_perc = height * 100.0
                
                label_name = "infection"
                
                result_item = {
                    "id": str(uuid.uuid4())[:8],
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": int(orig_w),
                    "original_height": int(orig_h),
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": float(x_perc),
                        "y": float(y_perc),
                        "width": float(w_perc),
                        "height": float(h_perc),
                        "rectanglelabels": [label_name],
                    },
                }
                task["predictions"][0]["result"].append(result_item)
                
        filtered_tasks.append(task)
        added_count += 1

    print(f"Added {added_count} new tasks from new_data.")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(filtered_tasks, f, indent=2)

    print(f"Successfully merged annotations. Saved {len(filtered_tasks)} total tasks to {json_path}")

if __name__ == "__main__":
    main()
