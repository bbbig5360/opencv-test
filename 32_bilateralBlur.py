import cv2
import numpy as np 


img = cv2.imread('data/images/salt-and-pepper.png')

dst = cv2.medianBlur( img, 5, 20 )

cv2.imshow('origin', img)
cv2.imshow('15',dst)

cv2.waitKey()
cv2.destroyAllWindows()