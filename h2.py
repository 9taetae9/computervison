import cv2
import numpy as np

# Function to preprocess the image
def preProcess(img):
    # Check if the image is already in grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        imgGray = img  # The image is already grayscale
    else:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return imgThreshold


# Function to find the biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# Function to reorder points for Warp Perspective
def reorder(myPoints):
    # Here, myPoints should be of the shape [4, 2]
    newPoints = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def detectGridSize(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find the lines on the chessboard
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough transform to find lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    # Count the number of lines detected
    if lines is not None:
        num_lines = len(lines)
        # Assuming a square chessboard, the number of squares on one side is
        # the square root of half the number of lines (since lines include both
        # vertical and horizontal)
        gridSize = int(np.sqrt(num_lines / 2))
    else:
        gridSize = 0  # If no lines are detected, return 0 or an appropriate default value
    
    return gridSize

def splitBoxes(img, gridSize):
    rows = np.vsplit(img, gridSize)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, gridSize)
        for box in cols:
            boxes.append(box)
    return boxes

# In your main logic, you would then use:
gridSize = detectGridSize(imgWarpColored)
if gridSize > 0:
    boxes = splitBoxes(imgWarpColored, gridSize)
else:
# Handle the case where the grid size isn't detected
# You might want to return an error or use a default value
    raise ValueError("Could not detect the grid size.")

# Function to get the grid size by detecting horizontal and vertical lines
def getGridSize(imgThreshold):
    # Detecting lines using HoughLinesP
    lines = cv2.HoughLinesP(imgThreshold, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
    
    if lines is None:
        return None
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) > abs(x2 - x1):  # vertical line
            vertical_lines.append(line)
        else:  # horizontal line
            horizontal_lines.append(line)
    
    # Sort and remove duplicates or too close lines to ensure we have unique counts
    def unique_lines(lines):
        unique = []
        for line in sorted(lines):
            if not unique or abs(line[0][0] - unique[-1][0][0]) > 10:  # Adjust threshold as needed
                unique.append(line)
        return unique

    horizontal_lines = unique_lines(horizontal_lines)
    vertical_lines = unique_lines(vertical_lines)

    # Counting the number of boxes between the first and last lines
    rows = len(horizontal_lines) - 1
    cols = len(vertical_lines) - 1
    
    # Ensure rows and cols are positive
    if rows > 0 and cols > 0:
        return rows, cols
    else:
        return None

    return rows, cols
# Function to pad the image to the nearest size divisible by the grid size
def padImageToGridSize(img, rows, cols):
    rowHeight = img.shape[0] // rows
    colWidth = img.shape[1] // cols

    # Calculate padding to add to make the image divisible by the grid size
    rowPadding = (rowHeight * rows) - img.shape[0]
    colPadding = (colWidth * cols) - img.shape[1]

    # Pad the bottom and right of the image
    imgPadded = cv2.copyMakeBorder(img, 0, rowPadding, 0, colPadding, cv2.BORDER_CONSTANT, None, value=0)
    return imgPadded

# Main function to process the image
def processImage(imgPath):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (450, 450))
    imgThreshold = preProcess(img)

    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, _ = biggestContour(contours)

    if biggest.size != 0:
        reordered = reorder(biggest)
        pts1 = np.float32(reordered)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (450, 450))
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        imgThreshold = preProcess(imgWarpGray)

        grid_size = getGridSize(imgThreshold)
        # Before calling padImageToGridSize, ensure that rows and cols are greater than zero.
        if grid_size and grid_size[0] > 0 and grid_size[1] > 0:
            print(f"Grid size detected: {grid_size}")
            imgPadded = padImageToGridSize(imgWarpGray, *grid_size)
            boxes = splitBoxes(imgPadded, *grid_size)
            print(f"Number of boxes: {len(boxes)}")
            return boxes
        else:
            print("Grid size not detected.")
            return None

    else:
        print("No grid found.")
        return None


# Using the function
boxes = processImage("check8_8_3.jpg")
if boxes:
    # You can now iterate over boxes to process each grid cell
    pass
