import cv2
import numpy as np

def determine_checkerboard_size(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    # Find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out very small contours which are most likely noise
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    if len(contours) >= 64:  # Expecting at least 64 squares for an 8x8 board
        return "8 x 8 (British/American rules)"
    elif len(contours) >= 100:  # Expecting at least 100 squares for a 10x10 board
        return "10 x 10 (International rules)"
    else:
        return "Checkerboard not recognized"

# Test with your images
img1 = cv2.imread('check8_8.jpg')
print(determine_checkerboard_size(img1))

img2 = cv2.imread('check10_10.jpg')
print(determine_checkerboard_size(img2))
