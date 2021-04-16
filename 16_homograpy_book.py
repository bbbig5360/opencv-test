import cv2
import numpy as np
# 호모그래피란 무엇인가?

# 4개 이상의 점 필요. 변환된 사진의 왜곡에 맞는 함수를 적용한다.
# 사람이 점을 찍어주었지만, 객체를 인식하는 인공지능이 있다면 자동화 가능? 

img_src = cv2.imread('data/images/book2.jpg',1)

point_img = np.array( [ 141,131, 480,159, 493,639, 64,601 ], dtype=float )
point_img = point_img.reshape(4,2)
print( point_img )

img_dst = cv2.imread('data/images/book1.jpg')
point_dst = np.array( [ 318,256, 534,372, 316,670, 73,473 ], dtype=float )
point_dst = point_dst.reshape(4,2)
print( point_dst )

h, status = cv2.findHomography(point_img, point_dst)
# h 가 변환에 사용될 3x3의 행렬이다

print(h)

img_output = cv2.warpPerspective(img_src, h, ( img_dst.shape[1], img_dst.shape[0] ) )
#                                   이미지 해상도 : 가로, 세로
cv2.imshow("SRC", img_src)
cv2.imshow('dst', img_dst)
cv2.imshow('warp', img_output)

cv2.waitKey()
cv2.destroyAllWindows()