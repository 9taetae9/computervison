import cv2
import numpy as np

#Step 1: Load the Image
image_path = 'check8_8_3.jpg'  # Update this path
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Step 2: Preprocess the Image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
adaptive_thresh = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

#Step 3: Edge Detection
edges = cv2.Canny(adaptive_thresh, 50, 150, apertureSize=3)

#Step 4: Line Detection
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
if lines is None:
    print("No lines were found")
    exit()

#Step 5: Find Intersections
# Define function to find intersections
def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return x0, y0

# Find all intersections
intersections = []
for i, line1 in enumerate(lines):
    for line2 in lines[i+1:]:
        intersections.append(intersection(line1, line2))

#Step 6: Geometric Filtering
# Filter points that form a square
# (This step requires geometric analysis of the points to determine which ones form a square)
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def find_corners(intersections):
    if len(intersections) < 4:
        return None
    intersections = np.array(intersections)
    
    # Compute the convex hull of the points
    hull = cv2.convexHull(intersections)
    
    # Filter points to get those that form a rectangle (4 corners)
    hull = hull[:, 0, :]  # Reshape hull array
    distances = [[distance(hull[i], hull[j]) for j in range(len(hull))] for i in range(len(hull))]
    distances = np.array(distances)
    
    corners = []
    for i in range(len(distances)):
        if len(corners) == 4:
            break
        for j in range(i + 1, len(distances)):
            if len(corners) == 4:
                break
            if np.isclose(distances[i][j], np.max(distances), atol=10):
                # Found the furthest pair, which should be diagonal points of the rectangle
                corners.append(hull[i])
                corners.append(hull[j])
                
    # Now find the other two points which form right angles with the diagonal
    if len(corners) == 2:
        pt1, pt2 = corners
        for pt in hull:
            if np.array_equal(pt, pt1) or np.array_equal(pt, pt2):
                continue
            if np.isclose(distance(pt, pt1) + distance(pt, pt2), distance(pt1, pt2), atol=10):
                corners.append(pt)
                
    if len(corners) < 4:
        return None
    
    # Sort the corners to ensure they are ordered consistently
    # [top-left, top-right, bottom-right, bottom-left]
    corners = sorted(corners, key=lambda x: x[0])  # Sort by x coordinate
    top_corners = sorted(corners[:2], key=lambda x: x[1])  # Sort by y coordinate
    bottom_corners = sorted(corners[2:], key=lambda x: x[1], reverse=True)
    corners = np.array(top_corners + bottom_corners)
    
    return corners

corners = find_corners(intersections)

#Step 7: Corner Refinement
# Harris Corner Detection
corners = cv2.cornerHarris(adaptive_thresh, 2, 3, 0.04)
# Refining corners using Harris corner detection
def refine_corners(harris_corners, rough_corners, window_size=5):
    refined_corners = []
    for corner in rough_corners:
        x, y = corner
        min_x = max(x - window_size, 0)
        max_x = min(x + window_size, harris_corners.shape[1])
        min_y = max(y - window_size, 0)
        max_y = min(y + window_size, harris_corners.shape[0])
        region = harris_corners[min_y:max_y, min_x:max_x]
        _, _, _, max_loc = cv2.minMaxLoc(region)
        refined_corners.append((min_x + max_loc[0], min_y + max_loc[1]))
    return np.array(refined_corners)

if corners is not None:
    corners = refine_corners(corners, corners)

# Additional step: Check if corners have been found and are valid
if corners is None or len(corners) != 4:
    print("Corners not detected properly.")
    exit()

# Ensure the corners are in the correct order
tl, tr, br, bl = corners

# Define the new points for the perspective transform
board_size = max(
    distance(tl, tr),
    distance(tr, br),
    distance(br, bl),
    distance(bl, tl)
)
board_size = int(board_size)
pts2 = np.float32([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]])

#Step 8: Crop the Chessboard
M = cv2.getPerspectiveTransform(corners.astype(np.float32), pts2)
dst = cv2.warpPerspective(image, M, (board_size, board_size))


cv2.imshow('Cropped Chessboard', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()