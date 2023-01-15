import cv2
import numpy as np






img = cv2.imread("resisgeez.jpg", 1)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_lower = np.array([41,57,78])
hsv_upper = np.array([145,255,255])
mask = cv2.inRange(img2, hsv_lower, hsv_upper)

kernel = np.ones((5, 5), np.uint8)
kernel2 = np.ones((40, 40), np.uint8)


image = cv2.erode(mask, kernel)
geez = cv2.dilate(image,kernel2)
finalgeez = cv2.erode(geez,np.ones((17, 17), np.uint8))

x, y, w, h = cv2.boundingRect(finalgeez)
rect1 = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,255,0),3) # not copying here will throw an error
cropped = img[y:y+h,x:x+w]
print("x:{0}, y:{1}, width:{2}, height:{3}".format(x, y, w, h))

hsv_lower2 = np.array([41,57,90])
hsv_upper2 = np.array([145,255,255])

img3 = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
img4 = cv2.cvtColor(cropped, cv2.COLOR_BGR2YCrCb)
mask2 = cv2.inRange(img3, hsv_lower, hsv_upper)
mask2 = cv2.bitwise_not(mask2)
retval, thresh = cv2.threshold(img3,80,255,cv2.THRESH_BINARY)
mask3 = cv2.inRange(img4, hsv_lower, hsv_upper)

x, y, w, h = cv2.boundingRect(cv2.erode(mask3, kernel))
rect1 = cv2.rectangle(cropped.copy(),(x,y),(x+w,y+h),(0,255,0),3) # not copying here will throw an error
sensitivity = 15
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
cropped2 = cropped[y:y+h,x:x+w]
hsv_lower2 = np.array([180,0,0])
hsv_upper2 = np.array([215,100,100])

img5 = cv2.cvtColor(cv2.GaussianBlur(cropped,(5,5),0), cv2.COLOR_BGR2Lab)
img6 = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

maske = cv2.inRange(img6, lower_white, upper_white)
maske = cv2.bitwise_not(maske)

mask = cv2.imread("vignetting.jpg")
image = img5
lowerRange = np.array([0, 135, 135], dtype="uint8")
upperRange = np.array([255, 160, 195], dtype="uint8")
mask = image[:].copy()

imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
imageRange = cv2.inRange(imageLab, lowerRange, upperRange)

mask[:, :, 0] = imageRange
mask[:, :, 1] = imageRange
mask[:, :, 2] = imageRange

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
faceLab = cv2.bitwise_and(image, mask)
# Convert to LAB color space
lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
lab_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2Lab)
lab_low = np.array([0,-128,-128])
lab_upper = np.array([100,127,127])
retval, thresh = cv2.threshold(lab_img,100,255,cv2.THRESH_BINARY)
YCrCb = cv2.cvtColor(cropped, cv2.COLOR_BGR2YCrCb)
ORANGE_MIN = np.array([0, 0, 200],np.uint8)
ORANGE_MAX = np.array([179, 255, 255],np.uint8)


hsv_img = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(lab_img)
ret, thresh_L = cv2.threshold(V, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
the_better_thresh = cv2.bitwise_not(thresh_L)
frame_threshed = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
# Split channels and lay out side-by-sise, Y on the left, Cr then Cb on the right
Y, Cr, Cb = cv2.split(YCrCb)
hstack = np.hstack((Y,Cr,Cb))
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(hstack)
#geezer = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
kernel3 = np.ones((13, 13), np.uint8)
maske = cv2.inRange(lab_img, lab_low, lab_upper)
processing = cv2.erode(thresh, kernel3)
x = []
y = []
for coords in np.argwhere(processing == 255):

    x.append(coords[0])
    y.append(coords[1])



x.sort()
first = x[0]
valting = x[-1]
for i in range(len(x)):
    if x[i] - x[i-1] > 1:
        print("ee")
        print(x[i])
        break





# Invert the vignetting (white no change, black increase brightness)
#inv_mask = (255 - lab_mask[:, :, 0])

# Add the vignetting contribution, clipping to the channel limit (0-255)
#lab_img[:, :, 0] = np.clip(lab_img[:, :, 0] + inv_mask, 0, 255)

# Back to RGB

# Resize for visualization
#res_img = cv2.resize(image, (w // 2, h // 2))
#res_dst = cv2.resize(result, (w // 2, h // 2))


contours, hierarchy = cv2.findContours(the_better_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
#elele = cv2.drawContours(cropped, contours, -1, (0, 255, 0), 3)
R = []
G = []
B = []
resis = []
print(cropped.shape)
from webcolors import rgb_to_name

cnts = contours
all_of_em = []
import time
for cnt in cnts:

        x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
        print(w,h)
        cv2.rectangle(cropped,(x,y),(x+w,y+h),(0,255,0),2)

        all_of_em.append(cropped[y:y+h,x:x+w])


import pandas as pd

#Training Data

import webcolors

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


colors = [["Orange","Peru"],["Brown","Maroon","Saddlebrown"],["Black"]]

#Predict
def rgb_to_hex(r, g, b):
  return ('{:X}{:X}{:X}').format(r, g, b)


print(rgb_to_hex(255, 165, 1))
#https://gist.githubusercontent.com/lunohodov/1995178/raw/80b1f09dd2a746db465be090a4f9893830153064/ral_standard.csv

bands = []
for band in all_of_em:
    xe,ye,_ = band.shape
    center_x,center_y = ((xe - 1) //2), ((ye-1)//2)
    color = band[center_x,center_y]
    R,G,B = color[2],color[1],color[0]
    requested_colour = (R, G, B)
    actual_name, closest_name = get_colour_name(requested_colour)
    for color in colors:
        print(color)
        for element in color:
            print(element)
            if closest_name == element.lower():
                bands.append(color[0])
            else:
                pass
    print(closest_name)
    #print(rgb_to_name((R,G,B), spec='css3'))
    print(rgb_to_hex(R,G,B))
print(bands)
#cv2.imshow('e', all_of_em[2])
#cv2.imshow('ee', all_of_em[3])
#cv2.imshow('eee', all_of_em[4])
#cv2.imshow('eeee', all_of_em[5])
#cv2.imshow('eeeee', all_of_em[6])

#mask2 = cv2.inRange(img5, hsv_lower, hsv_upper)
#dummy, img_cont = cv2.threshold(img2, thresh_level, 255, cv2.THRESH_BINARY)


# This will tell you which pixels are white
#print (np.where(is_white))

#contours, hierarchy = cv2.findContours(img_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#obj_index = contours.index(max(contours, key=len))
#print(obj_index)
#mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
#cv2.drawContours(mask, contours, obj_index, 255, -1) # Draw filled contour in mask
#out = np.zeros_like(img) # Extract out the object and place into output image
#out[mask == 255] = img[mask == 255]

# put mask into alpha channel of result
#result = img.copy()
#result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
#result[:, :, 3] = mask

#imfg2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


#mask2 = cv2.inRange(imfg2, hsv_lower, hsv_upper)

# save resulting masked image








cv2.waitKey(0)
