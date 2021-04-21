import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt

SET_WIDTH = int(600)
normalize_image = 1/255.0
resize_image_shape = (1024,512)

sample_img = cv2.imread('data4/images/example_02.jpg')
sample_img = imutils.resize(sample_img, width=SET_WIDTH)

blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image_shape, 0, swapRB=True, crop=False)

# ENET model 가져오기
cv_enet_model = cv2.dnn.readNet('data4/enet-cityscapes/enet-model.net')

cv_enet_model.setInput(blob_img)
# 1 : 1개의 이미지 입력
# 20 : 클래스의 개수
# 512, 1024 : 핼과 열의 갯수.
print(cv_enet_model)

cv_enet_model_output = cv_enet_model.forward()
print( cv_enet_model_output.shape )

# 라벨 이름을 로딩
label_values = open('data4/enet-cityscapes/enet-classes.txt').read().split('\n')
label_values = label_values[ :-1]
print(label_values)

IMG_OUTPUT_SHAPE_START = 1
IMG_OUTPUT_SHAPE_END = 4
classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

# 2. 모델의 아웃풋 20개 핼렬을, 하나의 행렬로 만든다.

class_map = np.argmax(cv_enet_model_output[0], axis=0)

CV_ENET_IMG_COLORS = open('data4/enet-cityscapes/enet-colors.txt').read().split('\n')
CV_ENET_IMG_COLORS = CV_ENET_IMG_COLORS[ : -1]

CV_ENET_IMG_COLORS = np.array([ np.array( color.split(',') ).astype('int') for color in CV_ENET_IMG_COLORS ] )


# 3.하나의 행렬을 이미지로 만든다.
# 각 픽셀별로, 클래스에 해당하는 숫자가 적힌 클래스 맵을,
# 각 숫자에 매핑되는 색깔(RGB)로 바꿔준다.
# 따라서 각 픽셀마다 색깔정보가 들어가면 된다.
mask_class_map = CV_ENET_IMG_COLORS[class_map]

# 리사이즈한다.

mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1],sample_img.shape[0]), interpolation=cv2.INTER_NEAREST )

# 그냥 더하면 255 를 넘어가므로 가중치를 두어 합한다.
cv_enet_model_output = ( ( 0.4 * sample_img ) + ( 0.6 * mask_class_map ) ).astype('uint8')

my_legend = np.zeros( ( len(label_values) * 25 , 300, 3 ) , dtype='uint8' )
for( i,(class_name, img_color) ) in enumerate( zip( label_values, CV_ENET_IMG_COLORS) ):
# 0, Unlabeled, [0,0,0] 형식으로 들어간다.
    color_info = [ int(color) for color in img_color ]
    # 혹시 몰라서 int로 바꿔줌.

    cv2.putText(my_legend, class_name, (5, (i*25)+17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2 )
    cv2.rectangle(my_legend, ( 100, (i*25) ), (300, (i*25)+25 ), tuple(color_info), -1  )


cv2.imshow('output', cv_enet_model_output)
cv2.imshow('origin', sample_img)
cv2.imshow('legend', my_legend)

cv2.waitKey()
cv2.destroyAllWindows()
