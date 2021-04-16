import cv2
import numpy as np 


img = cv2.imread('data/images/salt-and-pepper.png')

dst1 = cv2.medianBlur(img, 3 )
dst2 = cv2.medianBlur(img, 5 )
dst3 = cv2.medianBlur(img, 7 )
dst4 = cv2.medianBlur(img, 15 )
dst5 = cv2.medianBlur(img, 25 )


cv2.imshow('origin', img)
cv2.imshow('3',dst1)
cv2.imshow('5',dst2)
cv2.imshow('7',dst3)
cv2.imshow('15',dst4)
cv2.imshow('25',dst5)


cv2.waitKey()
cv2.destroyAllWindows()