import cv2
import numpy as np

# Masking the region of interest( ROI )
# 원하는 부분만 가져옵니다.
def reg_of_interest(image) :
    image_height = image.shape[0]
    polygons = np.array( [[ (200, image_height) , (1100, image_height), (550, 250) ]] )
    image_mask = np.zeros_like(image)
    cv2.fillPoly(image_mask, polygons, 255)
    masking_image = cv2.bitwise_and(image, image_mask)
    return masking_image

# Canny Edge Detection.
# 이미지를 Gray로 변환 후 가우시안 블러 사용. 다음에 엣지검출을 합니다.
def canny_edge(image) :
    gray_conversion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5), 0)
    canny_conversion = cv2.Canny(blur_conversion, 50, 150)
    return canny_conversion

# Hough Transform 적용하는 함수.
# https://opencv-python.readthedocs.io/en/latest/doc/25.imageHoughLineTransform/imageHoughLineTransform.html
def show_lines(image, lines) : 
    lines_image = np.zeros_like(image)
    if lines is not None :
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(lines_image,(x1,y1),(x2,y2),(0,0,255), 10 )
    return lines_image

# 비슷한 선이 여러개 나오니 1개만 나오도록 만들어 줍니다. ( 2개의 함수 이용. )
# -> y절편과 기울기를 평균화한다.
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1- intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameter = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameter[0]
        intercept = parameter[1]
        if slope < 0:
            right_fit.append((slope, intercept))
        else:
            left_fit.append((slope, intercept))
    left_fit_average =np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis =0)
    left_line =make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([[left_line, right_line]])


cap = cv2.VideoCapture('data3/test2.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if ret == False or (cv2.waitKey(25) & 0xFF==ord('q')):
        break

    canny_image = canny_edge(frame)
    roi_img = reg_of_interest(canny_image)
    lines = cv2.HoughLinesP( roi_img, 1, np.pi/180, 100, minLineLength=40, maxLineGap=5 )
    average_lines = average_slope_intercept(frame, lines)
    line_img = show_lines(frame, average_lines)
    combine_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    # 가중치는 원래   a x image1  + (a -1) x image2로 해야하지만 선을 잘 보기위해 1을 주었습니다.

    cv2.imshow('result', combine_img)

cap.release()
cv2.destroyAllWindows()
