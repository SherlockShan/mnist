import cv2

img = cv2.imread(r"C:\anaconda\sakura.jpg", 1)
# cv2.imshow('src', img)
imgInfo = img.shape
height= imgInfo[0]
width = imgInfo[1]
deep = imgInfo[2]

# 定义一个旋转矩阵
matRotate = cv2.getRotationMatrix2D((height*0.3, width*0.3), 45, 0.3) # mat rotate 1 center 2 angle 3 缩放系数

dst = cv2.warpAffine(img, matRotate, (height, width))

cv2.imshow('image',dst)
cv2.waitKey(0)
