import os
import json
import uuid
import cv2
from tqdm import tqdm
from ultralytics import YOLO


def calculate_overlap_ratio(boxA, boxB):
    # box is [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea == 0 or boxBArea == 0:
        return 0.0

    ratioA = interArea / boxAArea
    ratioB = interArea / boxBArea

    return max(ratioA, ratioB)


def merge_overlapping_boxes(boxes, class_ids, threshold=0.45):
    if not len(boxes):
        return [], []

    merged_any = True
    current_boxes = [list(b) + [c] for b, c in zip(boxes, class_ids)]

    while merged_any:
        merged_any = False
        new_boxes = []
        used = [False] * len(current_boxes)

        for i in range(len(current_boxes)):
            if used[i]:
                continue

            box = current_boxes[i].copy()
            used[i] = True

            for j in range(i + 1, len(current_boxes)):
                if used[j]:
                    continue

                if box[4] == current_boxes[j][4]:  # Same class
                    ratio = calculate_overlap_ratio(box[:4], current_boxes[j][:4])
                    if ratio > threshold:
                        # Merge them
                        box[0] = min(box[0], current_boxes[j][0])
                        box[1] = min(box[1], current_boxes[j][1])
                        box[2] = max(box[2], current_boxes[j][2])
                        box[3] = max(box[3], current_boxes[j][3])
                        used[j] = True
                        merged_any = True

            new_boxes.append(box)

        current_boxes = new_boxes

    return [b[:4] for b in current_boxes], [b[4] for b in current_boxes]


def main():
    # Paths
    model_path = os.path.join("models", "model_v4.pt")
    images_dir = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\clahe_leaves"
    output_json = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\label_studio_predictions_v5.json"

    # Prefix for Label Studio Local Storage
    IMAGE_URL_PREFIX = "/data/local-files/?d=whiteflies%5Cclahe_leaves%5C"

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    images = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not images:
        print(f"No images found in {images_dir}")
        return

    label_studio_tasks = []

    # We can use the model's classes dictionary, or provide a default if it doesn't exist
    classes = model.names if hasattr(model, "names") else {0: "infection"}

    for image_name in tqdm(images, desc="Predicting and Formatting for Label Studio"):
        img_path = os.path.join(images_dir, image_name)

        # We need original dimensions for Label Studio
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Run YOLO prediction
        results = model.predict(
            source=img_path, save=False, show=False, verbose=False, iou=0.9, conf=0.15
        )
        result = results[0]

        task = {
            "data": {"image": f"{IMAGE_URL_PREFIX}{image_name}"},
            "predictions": [{"result": []}],
        }

        # Extract boxes
        if len(result.boxes) > 0:
            # Get xyxyn (normalized x_min, y_min, x_max, y_max) for easier merging
            boxes = result.boxes.xyxyn.cpu().numpy()
            classes_ids = result.boxes.cls.cpu().numpy()

            # Merge overlapping boxes
            merged_boxes, merged_classes = merge_overlapping_boxes(
                boxes, classes_ids, threshold=0.45
            )

            for box, cls_id in zip(merged_boxes, merged_classes):
                x_min, y_min, x_max, y_max = box

                class_id = int(cls_id)
                label_name = classes.get(class_id, str(class_id))

                # Convert to percentages for Label Studio
                w_perc = float((x_max - x_min) * 100.0)
                h_perc = float((y_max - y_min) * 100.0)
                x_perc = float(x_min * 100.0)
                y_perc = float(y_min * 100.0)

                result_item = {
                    "id": str(uuid.uuid4())[:8],
                    "type": "rectanglelabels",
                    "from_name": "label",  # Ensure this matches your Label Studio labeling config
                    "to_name": "image",  # Ensure this matches your Label Studio labeling config
                    "original_width": int(orig_w),
                    "original_height": int(orig_h),
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
