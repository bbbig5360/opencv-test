import cv2
import numpy as np

img = cv2.imread('data/images/truth.png', 1)

laplacian = cv2.Laplacian(img, cv2.CV_32F, ksize=3, scale=1)


cv2.imshow('origin', img)
cv2.imshow('laplacian', laplacian)

cv2.waitKey()
cv2.destroyAllWindows()
