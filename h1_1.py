import cv2
import numpy as np

def determine_checkerboard_size(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Circle detection and removal
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=30, minRadius=5, maxRadius=40)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(gray, (i[0], i[1]), i[2], (127,127,127), -1)  # Overwrite circle with average checkerboard color

    
    # Image enhancement using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)
    
    # Thresholding
    _, threshed = cv2.threshold(enhanced_img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Finding contours
    contours, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour (after a certain size) is the checkerboard
    max_contour = max(contours, key=cv2.contourArea)
    
    # Applying perspective transform if the contour has 4 points (approximation)
    if len(max_contour) == 4:
        pts1 = np.array([max_contour[0], max_contour[1], max_contour[2], max_contour[3]], dtype="float32")
        side = max(img.shape)
        pts2 = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, matrix, (side, side))
    
    # Check for 8x8 checkerboard
    pattern_size = (7, 7)
    found, _ = cv2.findChessboardCorners(img, pattern_size)
    if found:
        return "8 x 8 (British/American rules)"
    
    # Check for 10x10 checkerboard
    pattern_size = (9, 9)
    found, _ = cv2.findChessboardCorners(img, pattern_size)
    if found:
        return "10 x 10 (International rules)"
    
    return "Checkerboard not recognized"
    



# Testing
image_paths = ['check8_8_1.jpg', 'check8_8_2.jpg','check8_8_3.jpg',
               'check10_10.jpg', 'check10_10_2.jpg','check12_12.jpg']  # replace with the paths to your images
for path in image_paths:
    print(determine_checkerboard_size(path))