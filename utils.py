import cv2
import numpy as np

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        print( data['img'])
        cv2.circle( data['img'], (x,y), 3, (0,0,255), 5, 16)
        cv2.imshow('image', data['img'])
        if len(data['points']) < 4:
            data['points'].append([x,y])

def get_four_points(img):
    data = {}
    data['img']=img.copy()
    data['points']=[]

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', mouse_handler, data)
    cv2.waitKey()
    # 유저가 마우스로 찍어둔 점을 float으로 바꿔야 한다.

    points = np.array(data['points'], dtype=float)
    return points
