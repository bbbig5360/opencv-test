import numpy as np
import imutils
import cv2

SET_WIDTH = int(600)
normalize_image = 1/255.0
resize_image_shape = (1024,512)

sample_img = cv2.imread('data4/images/example_02.jpg')
sample_img = imutils.resize(sample_img, width=SET_WIDTH)

blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image_shape, 0, swapRB=True, crop=False)

# ENET model 가져옵니다.
cv_enet_model = cv2.dnn.readNet('data4/enet-cityscapes/enet-model.net')

# 모델에 blob 데이터 입력합니다.
cv_enet_model.setInput(blob_img)
print(cv_enet_model)

# 출력값 받아옵니다.
cv_enet_model_output = cv_enet_model.forward()
print( cv_enet_model_output.shape )

# 개채의 이름과 넘버를 가져옵니다.
label_values = open('data4/enet-cityscapes/enet-classes.txt').read().split('\n')
label_values = label_values[ :-1]
# 끝에 '' 만 들어있는 데이터가 있어 버려줍니다.
print(label_values)

IMG_OUTPUT_SHAPE_START = 1
IMG_OUTPUT_SHAPE_END = 4
classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

# 모델의 아웃풋 20개 핼렬을, 하나의 행렬로 만듭니다.
class_map = np.argmax(cv_enet_model_output[0], axis=0)

# 클래스별 색깔을 가져옵니다.
CV_ENET_IMG_COLORS = open('data4/enet-cityscapes/enet-colors.txt').read().split('\n')
CV_ENET_IMG_COLORS = CV_ENET_IMG_COLORS[ : -1]
CV_ENET_IMG_COLORS = np.array([ np.array( color.split(',') ).astype('int') for color in CV_ENET_IMG_COLORS ] )


# E-net으로 추출한 Semantic Segmentation된 데이터를 클래스에 맞는 색을 넣어줍니다.
mask_class_map = CV_ENET_IMG_COLORS[class_map]

# 크기를 바꿔줍니다.
mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1],sample_img.shape[0]), interpolation=cv2.INTER_NEAREST )

# 그냥 더하면 255 를 넘어가므로 가중치를 두어 합한다.
cv_enet_model_output = ( ( 0.4 * sample_img ) + ( 0.6 * mask_class_map ) ).astype('uint8')


# 어떤 색깔이 어떤 클래스인지 표시하기위한 코드입니다.
my_legend = np.zeros( ( len(label_values) * 25 , 300, 3 ) , dtype='uint8' )
for( i,(class_name, img_color) ) in enumerate( zip( label_values, CV_ENET_IMG_COLORS) ):
    color_info = [ int(color) for color in img_color ]
    cv2.putText(my_legend, class_name, (5, (i*25)+17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2 )
    cv2.rectangle(my_legend, ( 100, (i*25) ), (300, (i*25)+25 ), tuple(color_info), -1  )

cv2.imshow('output', cv_enet_model_output)
cv2.imshow('origin', sample_img)
cv2.imshow('legend', my_legend)

cv2.waitKey()
cv2.destroyAllWindows()
