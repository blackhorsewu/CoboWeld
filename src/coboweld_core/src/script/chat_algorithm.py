import cv2
import numpy as np

# Load the point cloud data
point_cloud = np.load('point_cloud.npy')

# Convert the point cloud to grayscale image
img = cv2.cvtColor(point_cloud, cv2.COLOR_BGR2GRAY)

# Find the contours of the image
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through each contour
for contour in contours:
    # Find the convex hull of the contour
    hull = cv2.convexHull(contour, returnPoints=False)

    # Find the convexity defects of the contour
    defects = cv2.convexityDefects(contour, hull)

    # Loop through each convexity defect
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i][0]

        # Get the start, end, and far points of the defect
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculate the depth of the convexity defect
        depth = d / 256.0

        # If the depth is greater than a certain threshold, it is likely to be a seam or groove
        if depth > 10:
            # Draw a line connecting the start and end points of the defect
            cv2.line(img, start, end, (0, 0, 255), 2)

# Display the image with detected seams or grooves
cv2.imshow('Detected Seams/Grooves', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

