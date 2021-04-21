import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as p

SET_WIDTH = int(600)
normalize_image = 1/255.0
resize_image_shape = (1024,512)

cap = cv2.VideoCapture('data4/video/video.mp4')

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = cap.get(prop)
    print("[INFO] {} total frames in video.".format(total))
except:
    print('[INFO] could not determine number of frames in video')
    total = -1


# 모델 가져오기
cv_enet_model = cv2.dnn.readNet('data4/enet-cityscapes/enet-model.net')

# 라벨 이름을 로딩
label_values = open('data4/enet-cityscapes/enet-classes.txt').read().split('\n')
label_values = label_values[ :-1]

IMG_OUTPUT_SHAPE_START = 1
IMG_OUTPUT_SHAPE_END = 4

CV_ENET_IMG_COLORS = open('data4/enet-cityscapes/enet-colors.txt').read().split('\n')
CV_ENET_IMG_COLORS = CV_ENET_IMG_COLORS[ : -1]

CV_ENET_IMG_COLORS = np.array([ np.array( color.split(',') ).astype('int') for color in CV_ENET_IMG_COLORS ] )



while cap.isOpened():
    grapbbed, frame = cap.read()    
    
    video_frame = imutils.resize(frame, width=SET_WIDTH)

    # 모델에 넣을 이미지 형식 변환하기
    blob_img = cv2.dnn.blobFromImage( frame, normalize_image, resize_image_shape, 0, swapRB=True, crop=False )

    # 이미지 넣기
    cv_enet_model.setInput(blob_img)

    # 20장의 softmax값을 가진 이미지로 변환.
    cv_enet_model_output = cv_enet_model.forward()

    classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

    class_map = np.argmax(cv_enet_model_output[0], axis=0 )

    mask_class_map = CV_ENET_IMG_COLORS[class_map]
    # 해당 라벨을 3차원으로 만들어주는 것. 즉, 클래스별로 색깔이 입혀진 이미지가 된 것.

    mask_class_map = cv2.resize( mask_class_map, (video_frame.shape[1], video_frame.shape[0]), interpolation=cv2.INTER_NEAREST )
    cv_enet_model_output = ( (0.3 * video_frame) + (0.7 * mask_class_map ) ).astype('uint8')
    
    if grapbbed == True:
        cv2.imshow('video Semantic Segmentation', cv_enet_model_output)
        if cv2.waitKey(25) & 0xFF ==27:
            break
    else:
        break
    cap.release()

cv2.destroyAllWindows()

