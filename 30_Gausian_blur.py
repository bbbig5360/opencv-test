import cv2
import numpy as np 


img = cv2.imread('data/images/salt-and-pepper.png')

dst1 = cv2.GaussianBlur(img, (3,3), 50 )
dst2 = cv2.GaussianBlur(img, (5,5), 50 )
dst3 = cv2.GaussianBlur(img, (7,7), 50 )
dst4 = cv2.GaussianBlur(img, (15,15), 50 )
dst5 = cv2.GaussianBlur(img, (25,25), 50 )


cv2.imshow('origin', img)
cv2.imshow('3',dst1)
cv2.imshow('5',dst2)
cv2.imshow('7',dst3)
cv2.imshow('15',dst4)
cv2.imshow('25',dst5)


cv2.waitKey()
cv2.destroyAllWindows()