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
def reg_of_interest(image):
    image_height = image.shape[0]
    polygons = np.array([[ (200, image_height), (1100, image_height), (550,250)]])
    image_mask = np.zeros_like(image)
    cv2.fillPoly(image_mask, polygons, 255 )
    masking_image = cv2.bitwise_and(image, image_mask)
    return masking_image

# 6. Canny Edge Detection도 함수로 만들자.
def canny_edge(image):
    gray_conversion = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_conversion = cv2.GaussianBlur(gray_conversion, (5,5), 0)
    canny_conversion = cv2.Canny( blur_conversion, 50, 150 )
    return canny_conversion

# 7. Hough Transform 적용하는 함수.
# https://opencv-python.readthedocs.io/en/latest/doc/25.imageHoughLineTransform/imageHoughLineTransform.html

def show_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for i in range(len(lines)):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(lines_image,(x1,y1),(x2,y2),(0,0,255),10)
    return lines_image

# 선이 여러개 나오니 1개만 나오도록 최적화를 한다. -> y절편과 기울기를 평균화한다.
        
    
image = cv2.imread('data3/test_image.jpg')
lanelines_image = image.copy()

canny_conversion = canny_edge(lanelines_image)
cv2.imshow('canny',canny_conversion)
ROI_conversion = reg_of_interest(canny_conversion)
cv2.imshow('ROI',ROI_conversion)

lines = cv2.HoughLinesP(ROI_conversion, 1, np.pi/180, 50, minLineLength=50, maxLineGap=15)

lines_image = show_lines( lanelines_image, lines )

combine_image = cv2.addWeighted( lanelines_image, 0.8, lines_image, 1, 1 )

cv2.imshow('origin', image)
cv2.imshow('lines', lines_image)
cv2.imshow('combine',combine_image)


cv2.waitKey()
cv2.destroyAllWindows()