import cv2
import os
import matplotlib.pyplot as plt
from clahe_test import apply_clahe

images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground"
images = os.listdir(images_path)

# Pick an image
test_image = images[1]
img_path = os.path.join(images_path, test_image)

img = cv2.imread(img_path)
img = apply_clahe(img)

img = cv2.medianBlur(img, 27)

# Downscale for viewing if necessary
max_dim = 1200
if img.shape[0] > max_dim or img.shape[1] > max_dim:
    scale = max_dim / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

# Convert to different color spaces
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)


# Helper function to plot channels side-by-side
def plot_channels(img_array, title, channel_names):
    plt.figure(f"{title} Color Space", figsize=(15, 4))

    # Show the RGB image for reference in the first column
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Original (RGB)")
    plt.axis("off")

    # Plot each channel
    for i in range(3):
        plt.subplot(1, 4, i + 2)
        # We plot individual channels in grayscale to see intensity
        plt.imshow(img_array[:, :, i], cmap="gray")
        plt.title(f"{channel_names[i]} Channel")
        plt.axis("off")

    plt.tight_layout()


# 1. Plot RGB
plot_channels(img_rgb, "RGB", ["Red", "Green", "Blue"])

# 2. Plot HSV (Hue, Saturation, Value)
plot_channels(img_hsv, "HSV", ["Hue", "Saturation", "Value"])

# 3. Plot LAB (Lightness, A: Green-Red, B: Blue-Yellow)
plot_channels(img_lab, "LAB", ["L (Lightness)", "A (Green-Red)", "B (Blue-Yellow)"])

# 4. Plot YCrCb (Luma, Red-difference, Blue-difference)
plot_channels(img_ycrcb, "YCrCb", ["Y (Luma)", "Cr (Red-diff)", "Cb (Blue-diff)"])

# Display all figures
plt.show()
