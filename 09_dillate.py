import cv2

path = 'data/images/truth.png'

image = cv2.imread(path, cv2.IMREAD_COLOR)

cv2.imshow('origin', image)

dilationSize = 6
element = cv2.getStructuringElement( cv2.MORPH_RECT, 
                                        (2*dilationSize+1, 2*dilationSize+1),
                                        (dilationSize, dilationSize) )

imageDilate = cv2.dilate(image, element)
cv2.imshow('Dilation',imageDilate)

cv2.waitKey()
cv2.destroyAllWindows()