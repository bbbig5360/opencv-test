import cv2
import numpy as np

img = cv2.imread('data/images/candle.jpg',1)


scale = 200

ycbImage = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
cv2.imshow('crcb', ycbImage)

# 데이터 가공을 위해서 형변환. uint8 -> float 오버플로우 안 나도록.
ycbImage = np.float32(ycbImage)

# 채널 분리
Ychannel, Cr, Cb = cv2.split(ycbImage)

Ychannel = np.clip(Ychannel + scale, 0, 255)

# 타입 원래대로 돌려준다.
ycbImage = (cv2.merge([Ychannel, Cr, Cb])).astype('uint8')

# 화면 표시시 BGR로 변경.
ycbImage = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)

cv2.imshow('src', img)
cv2.imshow('ret',ycbImage)

# 하나의 윈도우에, 2개의 이미지를 옆으로(수평으로) 붙여서 표시하기.
img_all=np.hstack( [img, ycbImage] )

cv2.imshow('combined', img_all)

cv2.waitKey()
cv2.destroyAllWindows()