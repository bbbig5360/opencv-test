import cv2
import numpy as np

image = cv2.imread('data/images/mark.jpg')


imageLine = image.copy()
cv2.line(imageLine, (322,179), (400,183 ), (255,0, 0) , thickness=2, lineType=cv2.LINE_AA)
#                     start        end       color       선 두께
cv2.imshow('image line', imageLine)


imgCircle = image.copy()
cv2.circle(imgCircle, (350,200), 150, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
#                         중심   반경    색깔       두께          타입
cv2.imshow('image circle', imgCircle)


imgEllipse = image.copy()
cv2.ellipse(imgEllipse, (360,200), (100,170), 45, 0, 360, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
#         레퍼런스 참조해!      가장 먼 거리, 짧은거리  각도
cv2.ellipse(imgEllipse, (360,200), (100,170), 135, 0, 360, (0,0,255), thickness=2, lineType=cv2.LINE_AA)
cv2.imshow('image Ellipse', imgEllipse)


imgRect = image.copy()
cv2.rectangle(imgRect, (208,55), (450, 355), (255,0,0), thickness=4)
cv2.imshow('image Rectangle', imgRect)

imgText = image.copy()
cv2.putText(imgText, 'Mark Zuckerberg', (205,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow('image Text', imgText)

imgRT = image.copy()
cv2.rectangle(imgRT, (208,55), (450, 355), (255,0,0), thickness=4)
cv2.putText(imgRT, 'Mark Zuckerberg', (205,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow('image Rect & Text', imgRT)


cv2.waitKey()
cv2.destroyAllWindows()
