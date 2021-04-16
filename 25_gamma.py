import cv2
import numpy as np


img = cv2.imread('data/images/candle.jpg')

gamma=1.5

fullRange = np.arange(0,256)

lookupTable = np.uint8( 255 * np.power( (fullRange/255.0), gamma) )

print(lookupTable)

out = cv2.LUT(img, lookupTable)

combined = np.hstack([img,out])

cv2.imshow('combine', combined)

cv2.waitKey()

# 실패!