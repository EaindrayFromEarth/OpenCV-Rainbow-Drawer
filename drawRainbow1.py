import cv2
import numpy as np

frameWidth = 640
frameHeight = 480

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# Create a background subtractor to detect fingertips
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# Rainbow colors
myColorValues = [
    (255, 0, 0),   # Red
    (255, 165, 0),  # Orange
    (255, 255, 0),  # Yellow
    (0, 128, 0),   # Green
    (0, 0, 255),   # Blue
    (75, 0, 130),  # Indigo
    (128, 0, 128),  # Violet
]

myPoints = [[] for _ in range(len(myColorValues))]  # Create empty lists for each color
isDrawing = True  # Flag to indicate drawing or erasing

def findFingertips(img):
    # Apply background subtraction to detect fingertips
    fgmask = fgbg.apply(img)
    _, thresh = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0
    maxContour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            if area > maxArea:
                maxArea = area
                maxContour = cnt

    if maxContour is not None:
        M = cv2.moments(maxContour)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        return x, y
    else:
        return 0, 0

while True:
    # Capture a frame from the webcam
    success, img = cap.read()
    imgResult = img.copy()
    x, y = findFingertips(img)

    if x != 0 and y != 0:
        if isDrawing:
            # Draw all 7 color lines side by side
            for i, color in enumerate(myColorValues):
                myPoints[i].append((x + i * 20, y))
                for point in myPoints[i]:
                    cv2.circle(imgResult, (point[0], point[1]), 10, color, cv2.FILLED)

    # Show the result with the rainbow lines
    cv2.imshow("Result", imgResult)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        isDrawing = not isDrawing  # Toggle between drawing and erasing

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
