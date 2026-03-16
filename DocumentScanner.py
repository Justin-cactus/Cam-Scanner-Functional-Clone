import numpy as np
import cv2

path = "\Desktop\ECEN-447-LABS\FinalProject\Images"
image = cv2.imread(path + "\image1.jpg")

## Run Gaussian Blur

def GaussianBlur(image):
    # Gaussian Blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return blur

## Run Thresholding (Binary)

def Thresholding(image):
    # Thresholding
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

## Run Canny Edge Detection

def CanniedEdgeDetection(image):
    # Canny Edge Detection
    canny = cv2.Canny(image, 100, 200)
    return canny

## Find Contours

def FindContours(image):
    # Find Contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

## Four Point Transform

def FourPointTransform(image):
    # Four Point Transform
    pts = np.array([[0, 0], [0, image.shape[0]], [image.shape[1], image.shape[0]], [image.shape[1], 0]], dtype = "float32")
    pts = pts.reshape((-1, 1, 2))
    return pts

## Sharpen Image

def SharpenImage(image):
    # Sharpen Image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(image, -1, kernel)
    return sharpen

## Brighten Image

def BrightenImage(image):
    # Brighten Image
    bright = cv2.add(image, np.array([50]))
    return bright

## Transform

def Transform(image):
    # Transform
    Image = GaussianBlur(image)
    Image = Thresholding(Image)
    Image = CanniedEdgeDetection(Image)
    Image = FindContours(Image)
    Image = FourPointTransform(Image)
    Image = SharpenImage(Image)
    Image = BrightenImage(Image)
    return Image
## Testing

img = Transform(image)
cv2.imshow('image',img)
cv2.waitKey(0)




