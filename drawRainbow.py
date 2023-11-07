import cv2
import numpy as np

# Constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_CONTOUR_AREA = 1000

#This code uses a background subtractor 
# to identify the moving object (finger) 
# against a static background. 
# It may work better for finger tip detection in some situations.

# I dont like the way of detect skin color 
# within a specified range 
# in the HSV color space. 


# Create a VideoCapture object
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
cap.set(10, 150)

# Create a background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Rainbow colors
myColorValues = [
    (255, 0, 0),     # Red
    (255, 165, 0),   # Orange
    (255, 255, 0),   # Yellow
    (0, 128, 0),     # Green
    (0, 0, 255),     # Blue
    (75, 0, 130),    # Indigo
    (128, 0, 128),   # Violet
]

# Initialize points for each color
myPoints = [[] for _ in range(len(myColorValues))]

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    imgResult = img.copy()

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(img)

    # Apply morphological operations to clean up the mask
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y = approx[0][0]

            # Draw rainbow lines side by side
            for i, color in enumerate(myColorValues):
                myPoints[i].append([x + i * 20, y])

    # Draw the rainbow lines
    for i, points in enumerate(myPoints):
        for point in points:
            cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[i], cv2.FILLED)

    cv2.imshow("Result", imgResult)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
