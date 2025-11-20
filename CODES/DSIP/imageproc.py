# AIM: Perform gray level operations on images.

import cv2
import numpy as np

# Load the input grayscale image
image = cv2.imread('D:\CANVAS\SEM 5\DSIP\CODES\DSIP\ex1_3.png', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Function to perform image negation
def image_negation(input_image):
    negated_image = 255 - input_image
    return negated_image

# Function to perform image thresholding
def threshold_image(input_image, threshold_value):
    _, thresholded_image = cv2.threshold(input_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

# Function to perform image gamma correction
def image_gamma_correction(input_image, gamma):
    gamma_corrected_image = np.power(input_image / 255.0, gamma) * 255.0
    gamma_corrected_image = np.uint8(gamma_corrected_image)
    return gamma_corrected_image

# Perform gray level operations
negated_image = image_negation(image)
thresholded_image = threshold_image(image, 30)                                                        0)
gamma_corrected_image = image_gamma_correction(image, -1)

# Display the original and processed images
cv2.imshow("Original Image", image)
cv2.imshow("Negated Image", negated_image)
cv2.imshow("Thresholded Image", thresholded_image)
cv2.imshow("Gamma Corrected Image", gamma_corrected_image)

# Wait for a key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
