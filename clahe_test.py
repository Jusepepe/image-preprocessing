import cv2
import matplotlib.pyplot as plt


def apply_clahe(img):

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])
    clahe_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    return clahe_img


if __name__ == "__main__":
    image_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\leaves\image-7.jpg"
    image2_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground\image-7.png"
    image = cv2.imread(image_path)
    image2 = cv2.imread(image2_path)
    clahe_img = apply_clahe(image)
    clahe_img2 = apply_clahe(image2)
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    plt.title("CLAHE")
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Original 2")
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(clahe_img2, cv2.COLOR_BGR2RGB))
    plt.title("CLAHE 2")
    plt.show()
