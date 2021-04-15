# 특정 데이터만 잘라서 가져오기.
import cv2

source = cv2.imread('data/images/sample.jpg', 3)

scaleX = 0.6
scaleY = 0.6

scaleDown = cv2.resize(source, None, fx=scaleX, fy=scaleY, interpolation= cv2.INTER_LINEAR)

cv2.imshow('Original', source)
cv2.imshow('Scale Down', scaleDown)

scaleX = 1.6
scaleY = 1.6

scaleUp = cv2.resize(source, None, fx=scaleX, fy=scaleY, interpolation= cv2.INTER_LINEAR)

cv2.imshow('scale Up', scaleUp)

crop_img = source[ 150:200, 150:350 ]

cv2.imshow('Cropped Img', crop_img)

cv2.waitKey()
cv2.destroyAllWindows()
