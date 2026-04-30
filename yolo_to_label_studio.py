import os
import json
import uuid
import cv2
from tqdm import tqdm


def main():
    images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\clahe_leaves"
    labels_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\labels"
    output_json = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\label_studio_import.json"

    # Map your YOLO class IDs to Label Studio class names
    classes = {0: "infection"}

    # Prefix for Label Studio Local Storage.
    # Change this to match how your images are synced in Label Studio.
    # Common examples:
    # "/data/local-files/?d=foreground/"
    # "http://localhost:8081/foreground/"
    # Just the filename: ""
    IMAGE_URL_PREFIX = "/data/local-files/?d=whiteflies%5Cclahe_leaves%5C"

    label_studio_tasks = []

    images = [
        f
        for f in os.listdir(images_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for image_name in tqdm(images, desc="Formatting for Label Studio"):
        img_path = os.path.join(images_path, image_name)
        base_name = os.path.splitext(image_name)[0]
        txt_path = os.path.join(labels_path, f"{base_name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        task = {
            "data": {"image": f"{IMAGE_URL_PREFIX}{image_name}"},
            "predictions": [{"result": []}],
        }

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    label_name = classes.get(class_id, str(class_id))

                    # YOLO uses normalized center coordinates (0-1).
                    # Label Studio uses percentage coordinates for top-left corner (0-100).
                    w_perc = w_norm * 100.0
                    h_perc = h_norm * 100.0
                    x_perc = (x_center * 100.0) - (w_perc / 2.0)
                    y_perc = (y_center * 100.0) - (h_perc / 2.0)

                    result_item = {
                        "id": str(uuid.uuid4())[:8],
                        "type": "rectanglelabels",
                        "from_name": "label",  # Ensure this matches your Label Studio labeling config
                        "to_name": "image",  # Ensure this matches your Label Studio labeling config
                        "original_width": orig_w,
                        "original_height": orig_h,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": x_perc,
                            "y": y_perc,
                            "width": w_perc,
                            "height": h_perc,
                            "rectanglelabels": [label_name],
                        },
                    }
                    task["predictions"][0]["result"].append(result_item)

        label_studio_tasks.append(task)

    with open(output_json, "w") as f:
        json.dump(label_studio_tasks, f, indent=2)

    print(f"Exported {len(label_studio_tasks)} tasks to {output_json}")
    print("You can now import this JSON file into Label Studio.")


if __name__ == "__main__":
    main()
