import cv2
import numpy as np
from utils import get_four_points
# 호모그래피를 사용하는데 두번째 좌표를 구할 필요가 없다.

img_src = cv2.imread('data/images/times-square.jpg')
img_dst = cv2.imread('data/images/first-image.jpg')

dst_size = img_dst.shape

cv2.imshow('image', img_src)


# 원본 이미지로부터 마우스 클릭으로 4개의 점을 가져올 것이다.
# 새로 만들 이미지에서, 위의 원본 이미지 4개의 점과 매핑할 점을 잡아줘야한다.

points_src = np.array( [0,0,
                        img_dst.shape[1],0,
                        img_dst.shape[1],img_dst.shape[0], 
                        0, img_dst.shape[0] ], dtype=float)
points_src = points_src.reshape(4,2)

points_dst = get_four_points(img_src)

h, status = cv2.findHomography(points_src, points_dst)

img_dst = cv2.warpPerspective( img_dst, h, (img_src.shape[1],img_src.shape[0]) )

cv2.imshow('changed img', img_dst)


cv2.fillConvexPoly( img_src, points_dst.astype(int), 0 )

cv2.imshow('Img to 0', img_src)

sum_img = img_src + img_dst

cv2.imshow('changed photo', sum_img)

cv2.waitKey()
cv2.destroyAllWindows()