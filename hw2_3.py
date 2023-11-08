import cv2
import numpy as np

# Function to check if a set of four points form a rectangle
def is_rectangle(corners):
    # There should be 4 corners
    if len(corners) != 4:
        return False
    # Calculate distances between each pair of points
    dist = [np.linalg.norm(corners[i] - corners[(i+1) % 4]) for i in range(4)]
    # Opposite sides must be approximately equal in length
    if not np.isclose(dist[0], dist[2], atol=10) or not np.isclose(dist[1], dist[3], atol=10):
        return False
    # Calculate angles for all corners
    angles = []
    for i in range(4):
        p1, p2, p3 = corners[i-1], corners[i], corners[(i+1) % 4]
        dot_product = np.dot(p1-p2, p3-p2)
        norm_product = np.linalg.norm(p1-p2) * np.linalg.norm(p3-p2)
        angle = np.arccos(dot_product / norm_product) * 180 / np.pi
        angles.append(angle)
    # All angles must be approximately 90 degrees
    for angle in angles:
        if not np.isclose(angle, 90, atol=10):
            return False
    return True

# Load the Image
image_path = 'check8_8_3.jpg'  # Update this path
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess the Image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
adaptive_thresh = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the thresholded image
contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours based on the area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Go through the contours to find a rectangle that could be the checkerboard
for cnt in contours:
    # Approximate the contour to a polygon
    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Convert contour to a numpy array for further processing
    approx = np.squeeze(approx)

    # If the polygon has 4 points, we consider it as a candidate for our checkerboard
    if is_rectangle(approx):
        # If it's a rectangle, go ahead with the perspective transform
        # Ensure the corners are in the correct order (clockwise or counterclockwise)
        approx = approx[np.argsort(np.arctan2(approx[:, 1], approx[:, 0]))]

        # Define the new points for the perspective transform
        board_size = max([np.linalg.norm(approx[i] - approx[(i+1) % 4]) for i in range(4)])
        board_size = int(board_size)
        board_width = 300
        board_height = 300
        dst_points = np.array([
            [0, 0],
            [enlarged_size * board_width - 1, 0],
            [enlarged_size * board_width - 1, enlarged_size * board_height - 1],
            [0, enlarged_size * board_height - 1]
], dtype='float32')
        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(approx.astype(np.float32), dst_points)

        # Warp the image using the perspective transform
        dst = cv2.warpPerspective(image, M, (board_size, board_size))

        # Display the result
        cv2.imshow('Checkerboard', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
else:
    print("No checkerboard found.")
