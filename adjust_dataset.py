import os
import cv2
from tqdm import tqdm
from clahe_test import apply_clahe

images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\leaves"
images = os.listdir(images_path)

out_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\clahe_leaves"

if not os.path.exists(out_path):
    os.makedirs(out_path)

for image in tqdm(images):
    if not image.endswith(".jpg"):
        continue

    # Pick an image
    img_path = os.path.join(images_path, image)

    img = cv2.imread(img_path)
    clahe_img = apply_clahe(img)
    cv2.imwrite(os.path.join(out_path, image), clahe_img)
    print(f"Saved {image}")
