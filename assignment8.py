import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Load Images ------------------ #
# Main Image
img = cv2.imread(r'D:\kathan\python_codes\7th sem\CV lab\Comuter-Vision-Lab\Screenshot 2025-11-06 235048.png', cv2.IMREAD_GRAYSCALE)

# Reference image for histogram specification
ref = cv2.imread(r'D:\kathan\python_codes\7th sem\CV lab\Comuter-Vision-Lab\Screenshot 2025-11-06 235055.png', cv2.IMREAD_GRAYSCALE)

# Check if images loaded properly
if img is None or ref is None:
    print("Error: Could not load images. Check image paths.")
    exit()

# ------------------ Histogram Equalization ------------------ #
equalized_img = cv2.equalizeHist(img)

# ------------------ Histogram Specification (Matching) ------------------ #
# Function to match histograms
def hist_match(source, reference):
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, bins = np.histogram(reference.flatten(), 256, [0, 256])

    # Compute CDFs
    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()

    # Create lookup table
    lookup_table = np.zeros(256)
    for i in range(256):
        diff = np.abs(src_cdf[i] - ref_cdf)
        lookup_table[i] = np.argmin(diff)

    # Apply mapping
    matched = lookup_table[source]
    return matched.astype(np.uint8)

specified_img = hist_match(img, ref)

# ------------------ Display Results ------------------ #

plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

# Equalized Image
plt.subplot(2, 3, 2)
plt.title("Histogram Equalization")
plt.imshow(equalized_img, cmap='gray')
plt.axis('off')

# Specified Image
plt.subplot(2, 3, 3)
plt.title("Histogram Specification")
plt.imshow(specified_img, cmap='gray')
plt.axis('off')

# Histograms
plt.subplot(2, 3, 4)
plt.title("Original Histogram")
plt.hist(img.ravel(), 256, [0, 256], color='black')

plt.subplot(2, 3, 5)
plt.title("Equalized Histogram")
plt.hist(equalized_img.ravel(), 256, [0, 256], color='black')

plt.subplot(2, 3, 6)
plt.title("Specified Histogram")
plt.hist(specified_img.ravel(), 256, [0, 256], color='black')

plt.show()
