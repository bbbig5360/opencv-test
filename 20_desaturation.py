import cv2
import numpy as np

saturationScale = 0.01

img = cv2.imread('data/images/capsicum.jpg',1)

cv2.imshow('img', img)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_img = np.float32( hsv_img )

# 채널로 분리하는 방법
H, S, V = cv2.split( hsv_img )

# 유용한 함수! np.clip 함수를 이용해 최소, 최대값 고정한다. 0~255
# 0보다 작아도 0, 255보다 커도 255.

S = np.clip( S * saturationScale, 0, 255 )

# 나눈 채털을 하나로 합치는 함수.
hsv_img = cv2.merge( [H, S, V] )

# int로 변환한 데이터를 uint8로 변경한다.

hsv_img = np.uint8( hsv_img )

imgBgr = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

cv2.imshow('minus Saturation', imgBgr )


cv2.waitKey()
cv2.destroyAllWindows()