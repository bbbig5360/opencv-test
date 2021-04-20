import cv2
import numpy as np

# 이미지 가져오기
# image = cv2.imread('data3/test_image.jpg')

# cv2.imshow('origin', image)

# lanelines_image = image.copy()

# 2. 그레이 변환
# gray_conversion = cv2.cvtColor(lanelines_image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', gray_conversion)

# # 3. smoothing
# blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5), 0)
# cv2.imshow('smoothing', blur_conversion)

# # 4. Canny Edge Detection

# canny_conversion = cv2.Canny(blur_conversion, 50, 155)

# cv2.imshow('canny', canny_conversion)

# 5. Masking the region of interest( ROI )
def reg_of_interest(image) :
    image_height = image.shape[0]
    polygons = np.array( [[ (200, image_height) , (1100, image_height), (550, 250) ]] )
    image_mask = np.zeros_like(image)
    cv2.fillPoly(image_mask, polygons, 255)
    masking_image = cv2.bitwise_and(image, image_mask)
    return masking_image

# 6. Canny Edge Detection도 함수로 만들자.
def canny_edge(image) :
    gray_conversion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5), 0)
    canny_conversion = cv2.Canny(blur_conversion, 50, 150)
    return canny_conversion

# 7. Hough Transform 적용하는 함수.
# https://opencv-python.readthedocs.io/en/latest/doc/25.imageHoughLineTransform/imageHoughLineTransform.html

def show_lines(image, lines) : 
    lines_image = np.zeros_like(image)
    if lines is not None :
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(lines_image,(x1,y1),(x2,y2),(255,0,0), 10 )
    return lines_image

# 비슷한 선이 여러개 나오니 1개만 나오도록 만든다. ( 2개의 함수 이용. )
#                        -> y절편과 기울기를 평균화한다.
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


image = cv2.imread('data3/test_image.jpg')
lanelines_image = image.copy()

canny_conversion = canny_edge(lanelines_image)
roi_conversion = reg_of_interest(canny_conversion)

lines = cv2.HoughLinesP(roi_conversion, 1, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)

averaged_lines = average_slope_intercept(lanelines_image, lines)

lines_image = show_lines(lanelines_image, averaged_lines)

combine_image = cv2.addWeighted(lanelines_image, 0.8, lines_image, 1, 1)

# cv2.imshow('ori', lanelines_image)
# cv2.imshow("roi", lines_image)
# cv2.imshow("combined", combine_image)

cap = cv2.VideoCapture('data3/test2.mp4')


while cap.isOpened():
    ret, frame = cap.read()

    # frame 에 image 가 있다.
    canny_image = canny_edge(frame)
    roi_img = reg_of_interest(canny_image)
    lines = cv2.HoughLinesP( roi_img, 1, np.pi/180, 100, minLineLength=40, maxLineGap=5 )
    average_lines = average_slope_intercept(frame, lines)
    line_img = show_lines(frame, average_lines)

    combine_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    cv2.imshow('result', combine_img)

    print(ret)

    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
    
cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
