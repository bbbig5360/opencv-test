import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img_color = cv2.imread('data2/image.jpg')
# cv2.imshow('origin', img_color)


img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
# cv2.imshow('gray', img_gray)


img_copy = img_gray.copy()
# 값이 195 미만인 것들은 0으로 세팅.

img_copy[ img_copy[:,:] < 195 ] = 0

print(img_color.shape)
print(img_gray.shape)

# cv2.imshow('copyed',img_copy)

img = cv2.imread('data2/test_image.jpg')
print('Height = ', int(img.shape[0]), 'pixels')
print('widh = ', int(img.shape[1]), 'pixels')

# cv2.imshow('Self riving Car', img)

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow('SDC Gray', gray_img)

# hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv', hsv_img)

# H, S, V = cv2.split(hsv_img)
# H = hsv_img[:, :, 0]

# cv2.imshow('Hue', H)



sharp_kernel_1 = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
])
sharp_kernel_2 = np.array([
    [0,-1,0],
    [-1,7,-1],
    [0,-1,0]
])
sharpened_img1 = cv2.filter2D( gray_img, -1, sharp_kernel_1 )
sharpened_img2 = cv2.filter2D( gray_img, -1, sharp_kernel_2 )

# cv2.imshow('Gray', gray_img)
# cv2.imshow('sharp_img1', sharpened_img1)
# cv2.imshow('sharp_img2', sharpened_img2)


# blur_img = cv2.GaussianBlur(gray_img, (35,55), 1)
# cv2.imshow('blur', blur_img)


# sobel  edge detection
# x_sobel = cv2.Sobel(blur_img, cv2.CV_64F, 0, 1, ksize=7)
# y_sobel = cv2.Sobel(blur_img, cv2.CV_64F, 1, 0, ksize=7)
# cv2.imshow('x_sobel', x_sobel)
# cv2.imshow('y_sobel', y_sobel)
# cv2.imshow('xy_sobel', x_sobel + y_sobel)


# !! 라플라시안은 한번에 수직수평 다 잡는다. !!
# laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
# cv2.imshow('Lapla', laplacian)


# 라플라시안보다 캐니엣지가 더 좋다.
# low_threshold = 120
# high_threshold = 200

# canny_img = cv2.Canny(gray_img, low_threshold, high_threshold)
# cv2.imshow('Canny',canny_img)

# image = cv2.imread('data2/test_image2.jpg')
# cv2.imshow('origin', image)
# print(image.shape)
# M_rotation = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), 90, 0.5)

# rotated_img = cv2.warpAffine(image, M_rotation, (image.shape[1], image.shape[0]))

# cv2.imshow('rotated', rotated_img)

image = cv2.imread('data2/test_image3.jpg')
cv2.imshow('origin',image)

height = image.shape[0]
width = image.shape[1]

T_matrix = np.array( [
    [1,0,120],
    [0,1,-150]
    ], dtype='float32' )

print(T_matrix)

translation_img = cv2.warpAffine(image, T_matrix, (width, height))
cv2.imshow('trans', translation_img)

# resize - 확대, 축소.
resized_img = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
cv2.imshow('resized', resized_img)

cv2.waitKey()
cv2.destroyAllWindows()