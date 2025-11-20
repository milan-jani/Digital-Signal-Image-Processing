import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary fingerprint (grayscale)
#image = cv2.imread('Manual/exp_11_OH.png', 0)
image = cv2.imread('Manual/Exp_11_OH1.jpg', 0)

# --- Step 1: Remove small bright specks ---
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)

# --- Step 2: Fill thin dark cracks ---
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

# Optional smoothing
final = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

# --- Display results ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(opened, cmap='gray')
plt.title("After Opening (remove bright specks)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(closed, cmap='gray')
plt.title("After Closing (fill cracks)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(final, cmap='gray')
plt.title("Final Output")
plt.axis("off")

plt.tight_layout()
plt.show()
