import cv2
import numpy as np


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# image_path = r"C:\anaconda\sakura.jpg"

img = cv2.imread(r"C:\anaconda\sakura.jpg", 1)


imgInfo = img.shape
print( imgInfo )
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]

# 1 放大 缩小 2 等比例 非等比例
dstHeight = int(height * 0.3)
dstWeight = int(width * 0.3)

# 最近邻域插值 双线性插值 像素关系重采样 立方插值
dst = cv2.resize(img, (dstWeight,dstHeight))
cv2.imshow('image', dst)
# print(dst.shape)
# image = cv2.imread(image_path)
angle = 50
image_new = rotate_bound(dst, angle)
cv2.imshow('ww', image_new)
cv2.waitKey()
