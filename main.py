import cv2
import numpy as np

# Read the image
file_name = 'od.jpg'
full_path = 'od.jpg'
img = cv2.imread(full_path)

# Grayscale the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply filter and find edges
bfilter = cv2.bilateralFilter(gray, 11, 75, 75)  # Adjust the parameters
edged = cv2.Canny(bfilter, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Find number plate
roi = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        roi = approx
        break

# Check if a valid ROI is found
if roi is not None:
    roi = np.array([roi], np.int32)
    points = roi.reshape(4, 2)
    x, y = np.split(points, [-1], axis=1)

    (x1, x2) = (np.min(x), np.max(x))
    (y1, y2) = (np.min(y), np.max(y))
    number_plate = img[y1:y2, x1:x2]

    # Blur the image
    blurred_img = cv2.GaussianBlur(img, (51, 51), 30)

    # Create a mask for ROI and fill the ROI with white color
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(mask, roi, (255, 255, 255))

    # Create a mask for everywhere except ROI and fill with white color
    mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask

    # Combine all the masks and images
    result = cv2.bitwise_and(blurred_img, mask) + cv2.bitwise_and(img, mask_inverse)

    # Save and open the image
    cv2.imwrite('output/' + str(file_name[:-5]) + '_censored.jpg', result)  # Adjusted index for filename
    print('Successfully saved')
    saved = cv2.imread('output/' + str(file_name[:-5]) + '_censored.jpg')
    cv2.imshow('Saved', saved)

    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
else:
    print("Error: No valid ROI (license plate) found.")
