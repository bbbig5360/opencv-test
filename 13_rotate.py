import cv2

source = cv2.imread('data/images/sample.jpg', 1)
# 0 gray   1 color

# 회전의 중심 좌표 필요하다.
center = (source.shape[1]/2, source.shape[0]/2)

rotationAngle = 90
scaleFactor = 1

# cv2.getRotationMatrix2D( center, rotationAngle, scaleFactor )
#                           중심       회전 각        크기변환

rotationMatrix = cv2.getRotationMatrix2D( center, rotationAngle, scaleFactor )

print(rotationMatrix)

rotate_img = cv2.warpAffine( source, rotationMatrix, (source.shape[1], source.shape[0])  )

cv2.imshow('origin', source)
cv2.imshow('rotate', rotate_img)

cv2.waitKey()
cv2.destroyAllWindows()
