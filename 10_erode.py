import cv2

imgpath = 'data/images/truth.png'
image = cv2.imread(imgpath, cv2.IMREAD_COLOR)

# 이미지 깍아내기
dilationSize = 6

element = cv2.getStructuringElement( cv2.MORPH_RECT,
                                        (2*dilationSize+1, 2*dilationSize+1) )

imgEroded = cv2.erode(image, element)

cv2.imshow('origin', image)
cv2.imshow('erosion', imgEroded)

cv2.waitKey()
cv2.destroyAllWindows()