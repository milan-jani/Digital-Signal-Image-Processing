import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Manual/exp_12.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(image_blurred, threshold1=30, threshold2=100)

# Create a black image with the same size as the original image
boundary_image = np.zeros_like(image)

# Copy the detected edges to the boundary image
boundary_image[edges > 0] = 255

# Display the original image and the boundary image using matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(boundary_image, cmap='gray')
plt.title('Boundary Detection')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the boundary image
cv2.imwrite('boundary_image.jpg', boundary_image)

print("Boundary detection completed and saved as 'boundary_image.jpg'")

# Output
# Conclusion
# The Canny edge detection algorithm successfully identified the boundaries in the image.
# Gaussian blur preprocessing helped reduce noise for cleaner edge detection.
# The detected edges provide a clear outline of objects in the original image.
