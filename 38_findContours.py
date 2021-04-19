import cv2
import numpy as np
import random

threshold = 0

maxThreshold = 255
random.seed(12345)

def callback():
    # 캐니 엣지로, 엣지 검출하고,
    imgCanny = cv2.Canny(img, threshold, threshold*2, apertureSize=3)

    # 컨투어스 연결시킨다. edgess connection
    contours, heirarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    display = np.zeros( (imgCanny.shape[0], imgCanny.shape[1]) )

    for i in range(0, len(contours)):
        lineColor = (255,0,0)
        cnt = contours[i]
        cv2.drawContours(display, [cnt], -1, lineColor, 2)

    cv2.imshow('Contours', display / 255.0)

def updateTreshold(*args):
    global threshold
    threshold = args[0]
    callback()

img = cv2.imread('data/images/threshold.png', 0)

cv2.namedWindow('Contours', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Contours', img)

cv2.createTrackbar('Canny and Contours', 'Contours', threshold, maxThreshold, updateTreshold)


callback()

cv2.waitKey()
cv2.destroyAllWindows()