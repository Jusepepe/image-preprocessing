import os
import cv2
from tqdm import tqdm
from clahe_test import apply_clahe

images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground"
images = os.listdir(images_path)

out_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\adjusted_foreground"

if not os.path.exists(out_path):
    os.makedirs(out_path)

for image in tqdm(images):
    if not image.endswith(".png"):
        continue

    # Pick an image
    img_path = os.path.join(images_path, image)

    img = cv2.imread(img_path)

    median_img = cv2.medianBlur(img, 33)
    clahe_img = apply_clahe(median_img)
    cv2.imwrite(os.path.join(out_path, image), clahe_img)
    print(f"Saved {image}")
