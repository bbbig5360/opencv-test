import cv2
import numpy as np
from utils import get_four_points
# 호모그래피를 사용하는데 두번째 좌표를 구할 필요가 없다.

img_src = cv2.imread('data/images/times-square.jpg')
dst_img = cv2.imread('data/images/first-image.jpg')

cv2.imshow('image', img_src)

points_src = np.array( [0,0,
                        img_src.shape[1],0,
                        img_src.shape[1],img_src.shape[0], 
                        0, img_src.shape[0] ], dtype=float)



# points_dst = np.array( [0,0,
#                         dst_size[1],0,
#                         dst_size[1],dst_size[0], 
#                         0, dst_size[0] ], dtype=float)

points_dst = np.array([114, 219,
                       281, 360,
                       250, 446,
                        35, 338], dtype=float)

points_dst = points_dst.reshape(4,2)

h, status = cv2.findHomography( points_dst, points_src )

img_dst = cv2.warpPerspective( dst_img, h, (img_src.shape[1],img_src.shape[0]) )

cv2.imshow('change img', img_dst)


cv2.waitKey()
cv2.destroyAllWindows()