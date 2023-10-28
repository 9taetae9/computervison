import cv2
import numpy as np

def determine_checkerboard_size(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image")
        return 

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    # Inverse since findChessboardCorners works better with white squares
    inverse_adaptive = cv2.bitwise_not(adaptive_thresh)

    # Try to find the 10x10 checkerboard corners
    ret_10x10, corners_10x10 = cv2.findChessboardCorners(inverse_adaptive, (9,9))
    
    # If 10x10 not found, try 8x8
    if not ret_10x10:
        ret_8x8, corners_8x8 = cv2.findChessboardCorners(inverse_adaptive, (7,7))
        if ret_8x8:
            print("8 x 8 (British/American rules)")
            return 
        else:
            print("Checkerboard not recognized")
            return 
    else:
        print("10 x 10 (International rules)")

# Test
determine_checkerboard_size('check8_8.jpg')
determine_checkerboard_size('check10_10.jpg')
