import cv2 as cv
import numpy as np

# Initialize global variables
cnt = 0 
src_pts = np.zeros([4, 2], dtype=np.float32)

# Mouse callback function
def on_mouse(event, x, y, flags, param):
    global cnt, src_pts, src  # Add src to the global variables
    if event == cv.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32)
            cnt += 1

            # Draw a circle at the clicked point on the source image
            cv.circle(src, (x, y), 5, (0, 0, 255), -1)
            cv.imshow('check8_8_3.jpg', src)

        if cnt == 4:
            w = 400
            h = 400

            # Destination points for the perspective transform
            dst_pts = np.array([[0, 0],
                                [w - 1, 0],
                                [w - 1, h - 1],
                                [0, h - 1]], dtype=np.float32)
            
            # Calculate the perspective transform matrix
            pers_mat = cv.getPerspectiveTransform(src_pts, dst_pts)

            # Perform the perspective warp
            dst = cv.warpPerspective(src, pers_mat, (w, h))

            # Display the result
            cv.imshow('dst', dst)

# Load the source image
src = cv.imread('check8_8_3.jpg')

if src is None:
    print('Image load failed!')
    exit()

# Create a window and assign the callback function
cv.namedWindow('check8_8_3.jpg')
cv.setMouseCallback('check8_8_3.jpg', on_mouse, src)  # Pass src as the parameter to the callback

# Display the image and wait for a key press
cv.imshow('check8_8_3.jpg', src)
cv.waitKey(0)
cv.destroyAllWindows()
