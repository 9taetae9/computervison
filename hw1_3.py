import cv2
import numpy as np

def maskCircles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    
    # Use a relative size for minRadius and maxRadius based on image size
    minRadius = int(rows * 0.01)  # 1% of the image height
    maxRadius = int(rows * 0.10)  # 10% of the image height

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=150, param2=30,
                               minRadius=1, maxRadius=30)
    
    mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)  # Create a white mask
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2]+5, (0, 0, 0), -1)  # Increase radius slightly to ensure full coverage
    
    return mask

def needsWarp(biggest, img):
    img_aspect_ratio = img.shape[1] / img.shape[0]
    contour_width = np.linalg.norm(biggest[0] - biggest[1])
    contour_height = np.linalg.norm(biggest[1] - biggest[2])
    contour_aspect_ratio = contour_width / contour_height

    area_ratio = cv2.contourArea(biggest) / (img.shape[0] * img.shape[1])

    # If the aspect ratios are similar and the contour covers most of the image, no warp is needed
    if abs(img_aspect_ratio - contour_aspect_ratio) < 0.1 and area_ratio > 0.9:
        return False
    return True

# Function to preprocess the image and find edges
def is_chessboard(contour, img):
    # Check if the contour is quadrilateral
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return False

    # Check if the contour is large enough to be a chessboard
    if cv2.contourArea(approx) < 0.5 * img.shape[0] * img.shape[1]:
        return False
    
    return True

def preProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    threshold = cv2.adaptiveThreshold(blur, 255, 1, 1, 9, 2)
    return threshold

# Function to reorder points for Warp Perspective
def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

# Function to find the biggest contour
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if 50 < area :
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

# Function to draw contours on the image for debugging
def drawContours(img, contours):
    for cnt in contours:
        cv2.drawContours(img, [cnt], -1, (255, 0, 0), 3)
    return img

# Load the image
pathImage = "check8_8_8.jpg"
img = cv2.imread(pathImage)
heightImg, widthImg, _ = img.shape

# Preprocess the image
imgThreshold = preProcess(img)

# Apply the mask to the image
circle_mask = maskCircles(img)
masked_img = cv2.bitwise_and(imgThreshold, imgThreshold, mask=circle_mask)
# Find contours
contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



# Draw contours on the image for debugging
imgContours = img.copy()
imgContours = drawContours(imgContours, contours)
cv2.imshow("Contours", imgContours)  # This will show the image with drawn contours

# Find the biggest contour and assume it's the chessboard
biggest = biggestContour(contours)
if biggest.size != 0 and needsWarp(biggest, img):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest) # Prepare points for warp
    pts2 = np.float32([[0, 0], [widthImg-1, 0], [0, heightImg-1], [widthImg-1, heightImg-1]]) # Prepare points for warp
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    cv2.imshow("mask", masked_img)
    cv2.imshow("Warp Perspective", imgWarpColored)
else:
    imgWarpColored = img
    cv2.imshow("mask", masked_img)
    cv2.imshow("Warp Perspective", imgWarpColored)
    print("No Chessboard Found")

cv2.waitKey(0)
cv2.destroyAllWindows()
