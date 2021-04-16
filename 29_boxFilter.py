import cv2
import numpy as np

# boxfilter = 이미지를 부드럽게(블러링) 해주는 필터

img = cv2.imread('data/images/gaussian-noise.png')

dst1 = cv2.blur(img, (3,3) )
dst2 = cv2.blur(img, (5,5) )
dst3 = cv2.blur(img, (7,7) )


cv2.imshow('origin', img)
cv2.imshow('3 ',dst1)
cv2.imshow('5',dst2)
cv2.imshow('7',dst3)


cv2.waitKey()
cv2.destroyAllWindows()