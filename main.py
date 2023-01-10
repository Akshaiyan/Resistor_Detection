import cv2
import numpy as np



img = cv2.imread("resis.jpg", 1)
print(img.shape) # this should give you (img_h, img_w, 3)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

img_w, img_h = np.shape(img)[:2]
bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
thresh_level = bkg_level + 50

dummy, img_cont = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)


contours, hierarchy = cv2.findContours(img_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
obj_index = contours.index(max(contours, key=len))
print(obj_index)
contour_img = cv2.drawContours(img, contours, obj_index, (0,255,0), 3)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)


cv2.imshow('geez',contour_img)
cv2.waitKey(0)
