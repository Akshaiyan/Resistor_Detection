import cv2
import numpy as np

image = cv2.imread('geezers.jpg',1)
cv2.imshow('original',image)
blur = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow('blur',blur)
img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
img_blur = cv2.erode(image, (6,6))
edges = cv2.Canny(image=img_blur, threshold1=150, threshold2=200)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

theta = lines[0][0][1]

(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)

M = cv2.getRotationMatrix2D((cX, cY), 180*theta/3.1415926, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
rot_blur = cv2.GaussianBlur(rotated, (11,11), 0)
ret_img = cv2.cvtColor(rot_blur, cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(ret_img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)
S += 60
ret_img = cv2.merge((H, S, V))
ret_img = cv2.cvtColor(ret_img, cv2.COLOR_HSV2BGR)
cv2.imshow('edges',rotated)



cv2.waitKey(0)