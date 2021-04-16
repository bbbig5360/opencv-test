import cv2
import numpy as np

original = cv2.imread('data/images/girl.jpg')

img = original.copy()

# x축 피봇 포인트
originalValue = np.array([0,50,100,150,200,255])

# y축 포인트 : 빨간쪽과 파란쪽 각 부분의 포인트.

# 빨간 커브
rCurve = np.array([0,80,150,190,220,255])
# 파란 커브
bCurve = np.array([0,20,40,75,150,255])

fullrange = np.arange( 0, 255+1 )


bLUT = np.interp(fullrange, originalValue, rCurve)
rLUT = np.interp(fullrange, originalValue, bCurve)


# bChannel G rChannel = cv2.split(img)  과 같다.
rChannel = img[  : , : , 2  ]
rChannel = cv2.LUT(rChannel, rLUT)
img[ : , : , 2 ] = rChannel

bChannel = img[  : , : , 0  ]
bChannel = cv2.LUT(bChannel, bLUT)
img[ : , : , 0] = bChannel

combined = np.hstack([original, img])

cv2.imshow('combined', combined)

cv2.waitKey()
cv2.destroyAllWindows()