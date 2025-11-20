import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Manual/exp_12.png', 0)  # Load as grayscale

# Define the kernel (structuring element)
kernel = np.ones((10, 10), np.uint8)

# Perform Erosion
erosion = cv2.erode(image, kernel, iterations=10)

# Perform Dilation
dilation = cv2.dilate(image, kernel, iterations=1)

# Perform Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Perform Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the original image and the results using matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing')
plt.axis('off')

plt.tight_layout()
plt.show()

# Output
# Conclusion
# The morphological operations have demonstrated their effectiveness in processing the image.
# Erosion has removed small noise points, while dilation has helped in connecting disjoint structures.
# Opening has effectively removed small objects from the foreground, and closing has filled small holes in the foreground.
# These operations are crucial in various image processing tasks, including object detection and image segmentation.
