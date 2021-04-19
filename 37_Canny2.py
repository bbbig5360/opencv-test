import cv2
import numpy as np

threshold_high = 100
threshold_low = 50
threshold_max = 1000

apertureSizes = [3,5,7]
maxapertureIndex = 2
apertureIndex = 0

blurAmount=0
maxBlurAmount = 20

# track bar
def applyCanny():
    if blurAmount > 0:
        blurredSrc = cv2.GaussianBlur(src, (2*blurAmount+1, 2*blurAmount+1), 0)
    else:
        blurredSrc = src.copy()
    
    apertureSize = apertureSizes[apertureIndex]

    edges = cv2.Canny(blurredSrc, threshold_low, threshold_high, apertureSize=apertureSize)

    cv2.imshow('Edges', edges)

# low thrshold
def updateLowThreshold(*args):
    global threshold_low
    threshold_low = args[0]
    applyCanny()

def updateHighThreshold(*args):
    global threshold_high
    threshold_high = args[0]
    applyCanny()

def updateBlurAmount(*args):
    global blurAmount
    blurAmount = args[0]
    applyCanny()

def updateApertureIndex(*args):
    global apertureIndex
    apertureIndex = args[0]
    applyCanny()

src = cv2.imread('data/images/gaussian-noise.png', 0)

edges = src.copy()

cv2.imshow('Edges', src)
cv2.namedWindow('Edges', cv2.WINDOW_AUTOSIZE)

# low threshold's controler at trackbar
cv2.createTrackbar('Low Threshold', 'Edges', threshold_low, threshold_max, updateLowThreshold)

# high threshold's controler at trackbar
cv2.createTrackbar('High Threshold', 'Edges', threshold_high, threshold_max, updateHighThreshold)

# aperture를 트렉바에 붙인다.
cv2.createTrackbar('Aperture Size', 'Edges', apertureIndex, maxapertureIndex, updateApertureIndex)

# blur controler
cv2.createTrackbar('Blur', 'Edges', blurAmount, maxBlurAmount, updateBlurAmount)


cv2.waitKey()
cv2.destroyAllWindows()