import cv2
import numpy as np

# Region of Interest masking
# ROI 관심영역만 마스킹하는 것.

# img_color = cv2.imread('data2/test5.jpg')
# img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray image', img_gray)

# print(img_gray.shape)


# # blank = np.zeros( (img_gray.shape[0], img_gray.shape[1]) )
# blank = np.zeros_like(img_gray)

# ROI = np.array( [ [ (0,400), (300,250), (450,300), (700,427) ] ], dtype=np.int32 )
# mask = cv2.fillPoly(blank, ROI, 255)

# print(mask)

# masked_img = cv2.bitwise_and( img_gray, mask)

# cv2.imshow('blank', blank)
# cv2.imshow('masked', masked_img)

image_c = cv2.imread('data2/calendar.jpg')
image_g = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

image_Canny = cv2.Canny(image_g, 50, 200, apertureSize=3)
cv2.imshow('Canny', image_Canny)

lines = cv2.HoughLines(image_Canny, 1, np.pi / 180, 250)

for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 -1000*(a))

        cv2.line(image_c,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imshow('Hough Canny', image_c)


cv2.waitKey()
cv2.destroyAllWindows()