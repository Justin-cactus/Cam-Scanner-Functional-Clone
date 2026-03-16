import numpy as np
import cv2
import imutils

def order_points(pts):
	## Order the points so that they are in a clockwise order starting from top left
	rect = np.zeros((4, 2), dtype = "float32")
    
	s = pts.sum(axis = 1)
    
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
    
	diff = np.diff(pts, axis = 1)
    
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
    
	return rect

def four_point_transform(image, pts):
	## Call order_points function to order the points
	rect = order_points(pts)
	(topLeft, topRight, botRight, botLeft) = rect
    
	## Compute width of new image
	widthA = np.sqrt(((botRight[0] - botLeft[0]) ** 2) + ((botRight[1] - botLeft[1]) ** 2))
	widthB = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
    
	## Compute height of new image
	heightA = np.sqrt(((topRight[0] - botRight[0]) ** 2) + ((topRight[1] - botRight[1]) ** 2))
	heightB = np.sqrt(((topLeft[0] - botLeft[0]) ** 2) + ((topLeft[1] - botLeft[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
    
	## Compute the destination points for the perspective transform
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped

def Transform(image):
	## Create copy of image
	orig = image.copy()
    
	## Resize Image 
	ratio = image.shape[0] / 500.0
	image = imutils.resize(image, height = 500)
    
	## Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
	## Apply Gaussian Blur
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
	## Canny Edge Detection
	edged = cv2.Canny(gray, 30,200)
    
	## Find Contours
	contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
	for c in contours:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			screenCnt = approx
			break
            
	## Perform Perspective Transform with Four Points from Contour
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
	## Convert to Grayscale again
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
	## Apply thresholding to sharpen and make image more readable
	warped = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,10)
	return warped

## Read Image
image = cv2.imread('LetterForm.jpg')
try: 
	final = Transform(image)
	cv2.imwrite("Final.jpg", final)
	cv2.waitKey(0)
except:
	print("Error, no contour found")
