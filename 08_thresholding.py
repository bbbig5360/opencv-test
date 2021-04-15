import cv2

src = cv2.imread('data/images/threshold.png', 0)

# 구분하기 위한 값 설정
target = 0

# 위에서 설정한 값보다 큰 값들은, 모두 255로 색을 변경할 것.
maxValue = 255

cv2.imshow('origin', src)

th, dst = cv2.threshold( src, target, maxValue, cv2.THRESH_BINARY )

cv2.imshow('thresholded Image', dst)

cv2.waitKey()
cv2.destroyAllWindows()