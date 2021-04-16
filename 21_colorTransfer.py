import cv2
import numpy as np

src_img = cv2.imread('data/images/image1.jpg')
dst_img = cv2.imread('data/images/image2.jpg')

cv2.imshow('origin', src_img)
cv2.imshow('dst',dst_img)

output = dst_img.copy()

srcLab = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB)
dstLab = cv2.cvtColor(dst_img, cv2.COLOR_RGB2LAB)
outputLab = cv2.cvtColor(output, cv2.COLOR_RGB2LAB)

srcLab = srcLab.astype('float')
dstLab = dstLab.astype('float')
outputLab = outputLab.astype('float')

# srcLab = np.float32(srcLab) 도 같은기능을 한다.


# 채널 분리
srcL, srcA, srcB = cv2.split( srcLab )
dstL, dstA, dstB = cv2.split( dstLab )
outL, outA, outB = cv2.split( outputLab )

# 연산
outL = dstL - dstL.mean()
outA = dstA - dstA.mean()
outB = dstB - dstB.mean()

outL = outL * ( srcL.std()/dstL.std() )
outA = outA * ( srcA.std()/dstA.std() )
outB = outB * ( srcB.std()/dstB.std() )

outL = outL + srcL.mean()
outA = outA + srcA.mean()
outB = outB + srcB.mean()

# 적절한 범위 지정( 사진이므로 0 ~ 255 )
outL = np.clip(outL, 0, 255)
outA = np.clip(outA, 0, 255)
outB = np.clip(outB, 0, 255)

# 채널 합치기
outputLab = cv2.merge( [outL, outA, outB] )

# 다시 형변환해준다. ( 이미지는 8비트--1바이트-- 정수이기 때문에 )
outputLab = np.uint8(outputLab)

outputLab = cv2.cvtColor(outputLab, cv2.COLOR_LAB2BGR)


cv2.imshow('output', outputLab)


cv2.waitKey()
cv2.destroyAllWindows()