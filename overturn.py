import cv2

im = cv2.imread(r"C:\anaconda\lena512.bmp")
new_img = cv2.flip(im, flipCode=0)
cv2.imshow('image', new_img)
cv2.waitKey()
