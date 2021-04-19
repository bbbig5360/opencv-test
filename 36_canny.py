import cv2
import numpy as np

img = cv2.imread('data/images/gaussian-noise.pnga',0)

cv2.imshow('gray', img)

threshold_high = 100

threshold_low = 160  

result = cv2.Canny(img, threshold_high, threshold_low)

cv2.imshow('result', result)

cv2.waitKey()
cv2.destroyAllWindows()