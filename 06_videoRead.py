import cv2
import numpy as np


# FPS : Frame Per Second

# cap = cv2.VideoCapture('data/videos/chaplin.mp4')
cap = cv2.VideoCapture('data/videos/output.avi')

if cap.isOpened() == False:
    print('Error')

else:
    # 반복문이 필요한 이유!!! 
    # -> 비디오는 여러장의 사진으로 구성되어있기 떄문에 반복문 처리.
    while cap.isOpened():
        # 사진을 한장씩 가져온다.
        ret, frame = cap.read()

        # 제대로 된 사진이라면 화면에 출력.
        if ret == True:
            cv2.imshow('Frame', frame)

            # keyboard에서 esc키를 누르면 exit하라는 것.
            if cv2.waitKey(25) & 0xFF ==27:
                break

        else:
            break
    cap.release()


    cv2.destroyAllWindows()