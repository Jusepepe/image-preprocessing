import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from clahe_test import apply_clahe

images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground"

images = os.listdir(images_path)

test_image = images[1]
img_path = os.path.join(images_path, test_image)

img = cv2.imread(img_path)
img = apply_clahe(img)
img = cv2.medianBlur(img, 33)

# Downscale for viewing if necessary
max_dim = 1200
if img.shape[0] > max_dim or img.shape[1] > max_dim:
    scale = max_dim / max(img.shape[0], img.shape[1])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]

""" # Apply a sharpening filter to enhance texture differences before analysis
print("Applying sharpening filter...")
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gray = cv2.filter2D(gray, -1, sharpen_kernel) """

# --- 1. LBP Implementation using scikit-image ---
print("Computing Local Binary Patterns (LBP)...")
radius = 3
n_points = 10 * radius
lbp_img = local_binary_pattern(gray, n_points, radius, method="uniform")

print("Extracting texture busyness from LBP map...")
kernel_size = 9
mean_lbp = cv2.blur(lbp_img.astype(np.float32), (kernel_size, kernel_size))
mean_lbp2 = cv2.blur((lbp_img.astype(np.float32)) ** 2, (kernel_size, kernel_size))

# Variance = E[X^2] - (E[X])^2
lbp_variance = mean_lbp2 - mean_lbp**2
lbp_std = np.sqrt(np.maximum(lbp_variance, 0))

# Normalize the standard deviation to 0-255 for thresholding
lbp_std_norm = cv2.normalize(lbp_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# --- 2. GLCM Feature Extraction ---
print("Computing Gray Level Co-occurrence Matrix (GLCM)...")

# Reduce number of gray levels to speed up computation and reduce noise
# 256 levels is too sparse for GLCM, usually 8 or 16 levels are used
levels = 16
gray_quantized = (gray // (256 // levels)).astype(np.uint8)

# We compute GLCM on non-overlapping patches and resize the result
# to build a local texture map efficiently.
patch_size = 8

# Pad the image so its dimensions are multiples of patch_size
pad_h = (patch_size - gray_quantized.shape[0] % patch_size) % patch_size
pad_w = (patch_size - gray_quantized.shape[1] % patch_size) % patch_size
gray_padded = np.pad(gray_quantized, ((0, pad_h), (0, pad_w)), mode="reflect")

# Get dimensions of the patch grid
grid_h = gray_padded.shape[0] // patch_size
grid_w = gray_padded.shape[1] // patch_size

contrast_map = np.zeros((grid_h, grid_w), dtype=np.float32)
dissimilarity_map = np.zeros((grid_h, grid_w), dtype=np.float32)
homogeneity_map = np.zeros((grid_h, grid_w), dtype=np.float32)

print(f"Processing {grid_h * grid_w} patches for GLCM...")
for i in range(grid_h):
    for j in range(grid_w):
        patch = gray_padded[
            i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size
        ]

        # Compute GLCM for the patch. distances=[1], angles=[0 (horizontal), pi/2 (vertical)]
        glcm = graycomatrix(
            patch,
            distances=[1],
            angles=[0, np.pi / 2],
            levels=levels,
            symmetric=True,
            normed=True,
        )

        # Extract contrast property (variance of pixel differences)
        contrast = graycoprops(glcm, "contrast").mean()
        contrast_map[i, j] = contrast

        dissimilarity = graycoprops(glcm, "dissimilarity").mean()
        dissimilarity_map[i, j] = dissimilarity

        homogeneity = graycoprops(glcm, "homogeneity").mean()
        homogeneity_map[i, j] = homogeneity

# Resize the texture map back to the original image dimensions
texture_map = cv2.resize(
    contrast_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC
)
dissimilarity_map_resized = cv2.resize(
    dissimilarity_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC
)
homogeneity_map_resized = cv2.resize(
    homogeneity_map, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC
)

# Normalize the texture map to 0-255
glcm_contrast_norm = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX).astype(
    np.uint8
)
glcm_dissimilarity_norm = cv2.normalize(
    dissimilarity_map_resized, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)
glcm_homogeneity_norm = cv2.normalize(
    homogeneity_map_resized, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# --- 3. Plotting ---

# Figure 1: LBP Features
plt.figure("LBP Texture Analysis", figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(lbp_img, cmap="gray")
plt.title("Raw LBP Pattern Map")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(lbp_std_norm, cmap="gray")
plt.title("LBP Variance (Busyness)")
plt.axis("off")

plt.tight_layout()

# Figure 2: GLCM Features
plt.figure("GLCM Texture Analysis", figsize=(15, 5))

plt.subplot(1, 5, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(glcm_contrast_norm, cmap="jet")
plt.title("GLCM Contrast")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(glcm_dissimilarity_norm, cmap="jet")
plt.title("GLCM Dissimilarity")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(glcm_homogeneity_norm, cmap="jet")
plt.title("GLCM Homogeneity")
plt.axis("off")

plt.tight_layout()

plt.show()
