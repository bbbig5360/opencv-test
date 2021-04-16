import cv2
import numpy as np

input_tri = np.float32( [50,50, 100,100, 200,150] )
input_tri = input_tri.reshape(3,2)

# 변환 이미지의 3점 좌표

output_tri = np.float32( [70,76, 142,101, 272,136])
output_tri = output_tri.reshape(3,2)

print(input_tri)
print(output_tri)

warpMat = cv2.getAffineTransform(input_tri, output_tri)

print(warpMat)

# 두 사진의 피사체의 좌표를 받아 어떻게 변환했는지 식을 찾아내는 법
# Affine -> 평행이 유지된 상태에서 사진이 찌그러질 경우에.(삼각형)