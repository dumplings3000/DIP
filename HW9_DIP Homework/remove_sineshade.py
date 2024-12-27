import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('text-sineshade.tif', cv2.IMREAD_GRAYSCALE)

# Display original image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Apply Bilateral Filter to reduce noise while preserving edges
bilateral_filter = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Apply Gaussian Blur to further smooth the image
blurred_image = cv2.GaussianBlur(bilateral_filter, (5, 5), 0)

# Apply a morphological operation to enhance text
kernel = np.ones((3, 3), np.uint8)
morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)

# Apply adaptive thresholding to binarize the image
adaptive_thresh = cv2.adaptiveThreshold(morph_image, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Display the processed images
plt.subplot(1, 3, 2)
plt.title('Bilateral Filter')
plt.imshow(bilateral_filter, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Adaptive Thresholding')
plt.imshow(adaptive_thresh, cmap='gray')

# Save the final denoised image
cv2.imwrite('denoised_image.tif', adaptive_thresh)

plt.show()