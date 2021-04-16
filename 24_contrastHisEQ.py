import cv2
import numpy as np


img = cv2.imread('data/images/candle.jpg')

ycbImage = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

Ychannel, Cr, Cb = cv2.split(ycbImage)

print(Ychannel)

Ychannel = cv2.equalizeHist(Ychannel)

print(Ychannel)

ycbImage = cv2.merge([Ychannel, Cr, Cb])
ycbImage = cv2.cvtColor(ycbImage, cv2.COLOR_YCrCb2BGR)


cv2.imshow('convert', ycbImage)


cv2.waitKey()
cv2.destroyAllWindows()