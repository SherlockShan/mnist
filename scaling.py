import cv2
img = cv2.imread(r"C:\anaconda\sakura.jpg", 1)
imgInfo = img.shape
print( imgInfo )
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]

# 1 放大 缩小 2 等比例 非等比例
dstHeight = int(height * 0.4)
dstWeight = int(width * 0.4)

# 最近邻域插值 双线性插值 像素关系重采样 立方插值
dst = cv2.resize(img, (dstWeight,dstHeight))
print(dst.shape)
cv2.imshow('image', dst)
cv2.waitKey(0)
