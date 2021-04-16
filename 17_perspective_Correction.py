import cv2
import numpy as np
from utils import get_four_points
# 호모그래피를 사용하는데 두번째 좌표를 구할 필요가 없다.

img_src = cv2.imread('data/images/book1.jpg')
dst_size = (400, 250, 3)

img_dst = np.zeros(dst_size, dtype=np.uint8)

cv2.imshow('dst', img_dst)
cv2.imshow('image', img_src)


# 원본 이미지로부터 마우스 클릭으로 4개의 점을 가져올 것이다.
# 새로 만들 이미지에서, 위의 원본 이미지 4개의 점과 매핑할 점을 잡아줘야한다.

points_src = get_four_points(img_src)

points_dst = np.array( [0,0,
                        dst_size[1],0,
                        dst_size[1],dst_size[0], 
                        0, dst_size[0] ], dtype=float)

points_dst = points_dst.reshape(4,2)

h, status = cv2.findHomography(points_src, points_dst)

img_dst = cv2.warpPerspective( img_src, h, (dst_size[1],dst_size[0]) )

cv2.imshow('change img', img_dst)


cv2.waitKey()
cv2.destroyAllWindows()