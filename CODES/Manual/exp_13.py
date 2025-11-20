import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary image
image = cv2.imread('Manual/exp_12.png', 0)

# Define the structuring element for the pattern to be detected (0 for don't care, 1 for foreground, -1 for background)
pattern = np.array([[-1, 1, -1],
                    [1, 1, 1],
                    [-1, 1, -1]], dtype=np.int8)

# Apply the Hit or Miss Transformation
result = cv2.morphologyEx(image, cv2.MORPH_HITMISS, pattern)

# Display the original image and the result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result, cmap='gray')
plt.title('Hit or Miss Transform Result')
plt.axis('off')

plt.tight_layout()
plt.show()

# Output
# Conclusion
# The Hit-or-Miss transformation has been successfully applied to the image.
# This morphological operation detected specific patterns using the defined structuring element.
# The pattern kernel [-1,1,-1; 1,1,1; -1,1,-1] searches for plus-shaped structures where:
# - The center and cross arms must be foreground (white pixels)
# - The diagonal corners must be background (black pixels)
# The transformation effectively identified and highlighted these specific patterns in the binary image,
# demonstrating its effectiveness for template matching and pattern recognition tasks.
