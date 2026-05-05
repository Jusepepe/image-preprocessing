import cv2
import os
import numpy as np
from clahe_test import apply_clahe
from tqdm import tqdm
import re


def main():
    images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground"
    labels_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\labels\new_data"

    os.makedirs(labels_path, exist_ok=True)

    def is_valid_image(f):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            return False
        match = re.search(r"image-(\d+)", f)
        if match:
            num = int(match.group(1))
            return 877 <= num <= 997
        return False

    images = [f for f in os.listdir(images_path) if is_valid_image(f)]

    # New limits
    lower_lab = np.array([153, 117, 120])
    upper_lab = np.array([255, 135, 179])

    lower_hsv = np.array([0, 5, 150])
    upper_hsv = np.array([134, 88, 255])

    class_id = 0

    for image_name in tqdm(images, desc="Generating YOLO labels (877-997)"):
        img_path = os.path.join(images_path, image_name)
        original_img = cv2.imread(img_path)
        if original_img is None:
            continue

        img_h, img_w = original_img.shape[:2]

        clahe_img = apply_clahe(original_img.copy())
        blur_img = cv2.medianBlur(clahe_img, 11)
        clahe_img_lab = cv2.cvtColor(blur_img, cv2.COLOR_BGR2LAB)
        clahe_img_hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

        # Masks calculation
        lab_mask = cv2.inRange(clahe_img_lab, lower_lab, upper_lab)
        hsv_mask = cv2.inRange(clahe_img_hsv, lower_hsv, upper_hsv)

        result_mask_raw = cv2.bitwise_and(lab_mask, hsv_mask)

        # Apply morphology on the result mask
        mask_closed = cv2.morphologyEx(
            result_mask_raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2
        )
        result_mask = cv2.morphologyEx(
            mask_closed, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=3
        )

        contours, _ = cv2.findContours(
            result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        yolo_boxes = []

        def process_bbox(x, y, w, h):
            if w <= 0 or h <= 0:
                return
            sub_mask = result_mask[y : y + h, x : x + w]
            area = cv2.countNonZero(sub_mask)
            if area == 0:
                return

            bbox_area = w * h
            if bbox_area < 300:
                return

            # Stop if segment fills >75% of bbox, or if bbox is too small to divide usefully
            if area / bbox_area >= 0.65 or w < 5 or h < 5:
                # Calculate YOLO normalized coordinates
                x_center = (x + w / 2.0) / img_w
                y_center = (y + h / 2.0) / img_h
                w_norm = w / float(img_w)
                h_norm = h / float(img_h)

                # Clip values strictly between 0 and 1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                yolo_boxes.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )
            else:
                w1, h1 = w // 2, h // 2
                w2, h2 = w - w1, h - h1
                process_bbox(x, y, w1, h1)
                process_bbox(x + w1, y, w2, h1)
                process_bbox(x, y + h1, w1, h2)
                process_bbox(x + w1, y + h1, w2, h2)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            process_bbox(x, y, w, h)

        # Write to txt file
        base_name = os.path.splitext(image_name)[0]
        txt_path = os.path.join(labels_path, f"{base_name}.txt")

        with open(txt_path, "w") as f:
            if yolo_boxes:
                f.write("\n".join(yolo_boxes) + "\n")
            else:
                # Create an empty file to indicate no objects found
                f.write("")

    print(f"Finished processing {len(images)} images. Labels saved to {labels_path}")


if __name__ == "__main__":
    main()
