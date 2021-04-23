import numpy as np
import cv2

SET_WIDTH = int(600)
normalize_image = 1/255.0
resize_image_shape = (1024,512)

normalize_image = 1 / 255.0
resize_image_shape = (1024, 512)

sv = cv2.VideoCapture('data4/video/video.mp4')
cv_enet_model = cv2.dnn.readNet('data4/enet-cityscapes/enet-model.net')

CV_ENET_IMG_COLORS = open('data4/enet-cityscapes/enet-colors.txt').read().split('\n')
CV_ENET_IMG_COLORS = CV_ENET_IMG_COLORS[ : -1]
CV_ENET_IMG_COLORS = np.array([ np.array( color.split(',') ).astype('int') for color in CV_ENET_IMG_COLORS ] )

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = sv.get(prop)
    print("[INFO] {} total frames in video.".format(total))
except:
    print('[INFO] could not determine number of frames in video')
    total = -1

while sv.isOpened():
    grapbbed, frame = sv.read()
    if grapbbed == False:
        break 

    video_frame = imutils.resize(frame, width=SET_WIDTH)

    blob_img = cv2.dnn.blobFromImage( frame, normalize_image, resize_image_shape, 0, swapRB=True, crop=False )
    
    cv_enet_model.setInput(blob_img)
    cv_enet_model_output = cv_enet_model.forward()

    (classes_num, height, width) = cv_enet_model_output.shape[1:4]
    class_map = np.argmax(cv_enet_model_output[0], axis=0 )

    mask_class_map = CV_ENET_IMG_COLORS[class_map]

    mask_class_map = cv2.resize( mask_class_map, (video_frame.shape[1], video_frame.shape[0]), interpolation=cv2.INTER_NEAREST )
    cv_enet_model_output = ( (0.3 * video_frame) + (0.7 * mask_class_map ) ).astype('uint8')
    
    cv2.imshow('Frame', cv_enet_model_output)

    if cv2.waitKey(25) & 0xFF ==27:
        break

sv.release()
cv2.destroyAllWindows()