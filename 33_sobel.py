import cv2
import numpy as np

img=cv2.imread('data/images/truth.png')

sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)

cv2.imshow('origin', img)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)

cv2.imshow('sobelxy', sobelx + sobely)


cv2.waitKey()
cv2.destroyAllWindows()