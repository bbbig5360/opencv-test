import cv2 as cv

imageName = 'data/images/sample.jpg'

# opencv image open

image = cv.imread(imageName, cv.IMREAD_COLOR)

if image is None:
    print("can't open image")

print(image)

print(image.shape)

# Gray Scale Image : 1개의 행렬로 만들고, 0~255까지의 숫자로 채워진
# 행렬로 변환한 이미지.

grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

cv.imshow("image", image)
cv.imshow("gray scale", grayImage)

# 위의 코드는 이미지를 화면에 표시하고 바로 종료된다.
# 이 파일 자체를 cpu가 실행 후 끝냈기 떄문.

# 따라서, 위의 함수를 실행시켜서 눈으로 보기위해서는
# 다음처럼 코드를 작성한다.

cv.waitKey()
cv.destroyAllWindows()