import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import numpy as np

# -------------------------------------------------------------------
# --- 1. SET YOUR IMAGE PATH HERE ---
# -------------------------------------------------------------------
# Replace this with the path to one of your images from Teams
image_path = 'your_image.jpg' 
# -------------------------------------------------------------------

# Load the image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    print("Please make sure the path is correct and the file is a valid image.")
else:
    # Convert to grayscale for feature detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- 2. SIFT (Scale-Invariant Feature Transform) ---
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)
    
    # Draw keypoints on the original (color) image
    # DRAW_RICH_KEYPOINTS flags draws a circle with scale and orientation
    img_sift = cv2.drawKeypoints(image, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # --- 3. HOG (Histogram of Oriented Gradients) ---
    
    # Compute HOG features and the visualization image
    # Note: HOG doesn't find 'keypoints' in the same way SIFT does.
    # It creates a dense feature vector for the entire image (or patches).
    # The 'visualize=True' parameter generates an image that shows
    # the gradients.
    fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    
    # Rescale the HOG image for better contrast
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    # --- 4. Compare and Visualize ---
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    ax = axes.ravel()
    
    # Plot Original Image
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Matplotlib expects RGB
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Plot SIFT Features
    ax[1].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
    ax[1].set_title('SIFT Keypoints')
    ax[1].axis('off')
    
    # Plot HOG Features
    ax[2].imshow(hog_image_rescaled, cmap='gray')
    ax[2].set_title('HOG Visualization')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()

    print(f"SIFT found {len(keypoints_sift)} keypoints.")
    print(f"HOG feature descriptor shape: {fd.shape}")