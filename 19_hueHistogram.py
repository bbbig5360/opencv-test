import cv2
import numpy as np

img = cv2.imread('data/images/sample.jpg',1 )

cv2.imshow('img', img)

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', gray_img)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv_img)

print('hsv', hsv_img)

# hue          hsv_img[0]
# saturation   hsv_img[1]
# value        hsv_img[2]

hsv_img[2] = hsv_img[2] - 30

hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
cv2.imshow('value minus BGR', hsv_img)

cv2.waitKey()
cv2.destroyAllWindows()