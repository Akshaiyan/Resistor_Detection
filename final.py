import cv2
import numpy as np
import time
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

image = cv2.imread("eeee.jpg",1)


def brighten(val,image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    V += val
    ret_img = cv2.merge((H, S, V))
    ret_img = cv2.cvtColor(ret_img,cv2.COLOR_HSV2BGR)
    return ret_img


def prep_img(image):

    blur = cv2.GaussianBlur(image, (15, 15), 0)
    ret_img = brighten(15,blur)

    ret_img = cv2.cvtColor(ret_img,cv2.COLOR_BGR2HSV)
    proc = cv2.inRange(ret_img, (0,0,0), (179,255,100))
    kernel = np.ones((20, 20), np.uint8)


    proc = cv2.erode(proc, kernel)
    return proc





def band_order(cropped,crop2):
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cont_list = []
    ref_list = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cont_list.append(crop2[y:y + h, x:x + w])
        ref_list.append(crop2[y:y + h, x:x + w])
    y,x = cropped.shape
    temp = 0
    for i in range(1,x):
        try:
            cont = ref_list.index(i)
            cont_list[temp] = ref_list[cont]
            temp += 1
        except:
            pass
    #print(cont_list)
    return cont_list

img = prep_img(image)
x, y, w, h = cv2.boundingRect(img)

w = w - 10
h = h//2
cropped = img[y:y+h,x:x+w]
crop2 = image[y:y+h,x:x+w]
cropped_img = image[crop2.shape[0]//2:crop2.shape[0]]

procting = brighten(25,crop2)
blur = cv2.GaussianBlur(procting, (15, 15), 0)
blur = brighten(15,blur)
procting2 = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
H,S,V = cv2.split(procting)


for i in range(0,179):
    eingeez = 0
    proc = cv2.inRange(procting2, (0, 0, 0), (179, 255, i))
    proc = cv2.bitwise_not(proc)
    contours, hierarchy = cv2.findContours(proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in contours:
        eingeez += 1
    if eingeez == 4:
        break

    cv2.imshow('rpoc',proc)



bands = band_order(cropped,crop2)
for cnt in contours:
    y,x,_ = cnt.shape

    b, g, r = (crop2[y//2, x//2])
    print(r,g,b)
    requested_colour = (b, g, r)
    actual_name, closest_name = get_colour_name(requested_colour)
    print(closest_name)
blur = cv2.GaussianBlur(crop2, (35, 35), 0)


cv2.imshow("fatgeez",procting)


#x,y,w,h = cv2.boundingRect(img)
cv2.waitKey(0)
