import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def apply_clahe(img):

    # Downscale for viewing if necessary
    max_dim = 2500
    if img.shape[0] > max_dim or img.shape[1] > max_dim:
        scale = max_dim / max(img.shape[0], img.shape[1])
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
    clahe_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    return clahe_img


if __name__ == "__main__":
    images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground"
    images = os.listdir(images_path)

    for image in images[:25]:
        # Pick an image
        img_path = os.path.join(images_path, image)

        img = cv2.imread(img_path)

        median_img = cv2.medianBlur(img, 33)

        """ print("Applying sharpening filter...")
        blur_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        smooth_image = cv2.filter2D(clahe_img, -1, sharpen_kernel) """

        sharpened_image = cv2.addWeighted(img, 1.5, median_img, -0.5, 0)
        clahe_img = apply_clahe(sharpened_image)

        # Gradient features
        gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        phase = np.arctan2(dy, dx)
        phase = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        plt.figure(figsize=(15, 5))
        plt.subplot(3, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(3, 2, 2)
        plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
        plt.title("CLAHE Enhanced Image")
        plt.axis("off")
        plt.subplot(3, 2, 3)
        plt.imshow(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))
        plt.title("Median Filtered Image")
        plt.axis("off")
        plt.subplot(3, 2, 4)
        plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
        plt.title("Sharpened Image")
        plt.axis("off")
        plt.subplot(3, 2, 5)
        plt.imshow(magnitude, cmap="gray")
        plt.title("Magnitude")
        plt.axis("off")
        plt.subplot(3, 2, 6)
        plt.imshow(phase, cmap="gray")
        plt.title("Phase")
        plt.axis("off")
        plt.show()
