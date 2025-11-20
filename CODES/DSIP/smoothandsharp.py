import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('c:/Users/DJ/Downloads/smoothening data.png')

# --- Gaussian Smoothing ---
kernel_size = (5, 5)
sigma = 1.5
gaussian_kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
gaussian_kernel = np.outer(gaussian_kernel, gaussian_kernel)
smoothed_gaussian = cv2.filter2D(image, -1, gaussian_kernel)

# --- Sharpening (simple kernel) ---
sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
sharpened_simple = cv2.filter2D(image, -1, sharpening_kernel)

# Display: Original, Gaussian Smoothed, Sharpened
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('Gaussian Smoothed')
plt.imshow(cv2.cvtColor(smoothed_gaussian, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Sharpened')
plt.imshow(cv2.cvtColor(sharpened_simple, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# --- Averaging Filter ---
avg_kernel_size = (5, 5)
avg_kernel = np.ones(avg_kernel_size, dtype=np.float32) / (avg_kernel_size[0] * avg_kernel_size[1])
smoothed_average = cv2.filter2D(image, -1, avg_kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Averaged (Smoothed)')
plt.imshow(cv2.cvtColor(smoothed_average, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# --- Median Filter ---
median_kernel_size = 5  # must be odd
smoothed_median = cv2.medianBlur(image, median_kernel_size)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Median (Smoothed)')
plt.imshow(cv2.cvtColor(smoothed_median, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# --- Sharpening using Laplacian Kernel on Median Smoothed Image ---
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]], dtype=np.float32)
sharpened_laplacian = cv2.filter2D(smoothed_median, -1, laplacian_kernel)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Laplacian (Sharpened)')
plt.imshow(cv2.cvtColor(sharpened_laplacian, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
