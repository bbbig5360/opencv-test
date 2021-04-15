import cv2 
import numpy as np

img_path = "data/images/opening.png"

image = cv2.imread(img_path, 0)

cv2.imshow("origin", image)


openingSize = 3

element = cv2.getStructuringElement( cv2.MORPH_ELLIPSE,
                                        (2*openingSize+1, 2*openingSize+1))

imageOpened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element, iterations=3)

cv2.imshow('opened', imageOpened)
                                    


cv2.waitKey()
cv2.destroyAllWindows()