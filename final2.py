import cv2
import numpy as np
from collections import *
img1 = cv2.imread('geezers.jpg', 1)

hue, sat, val = img1[:,:,0], img1[:,:,1], img1[:,:,2]
a = np.mean(img1[:,:,2])
print('Image 1 V channel average=', a)
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    image = img1
    # cv2.imshow('original',image)
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # cv2.imshow('blur',blur)
    img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_blur = cv2.erode(image, (6, 6))
    edges = cv2.Canny(image=img_blur, threshold1=150, threshold2=200)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

    theta = lines[0][0][1]

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), 180 * theta / 3.1415926, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    rot_blur = cv2.GaussianBlur(rotated, (11, 11), 0)
    ret_img = cv2.cvtColor(rot_blur, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(ret_img, cv2.COLOR_BGR2HSV)

    ret_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2LAB)
    (l, a, b) = cv2.split(ret_img)
    L, A, B = cv2.split(ret_img)

    ret, thresh_L = cv2.threshold(B, 200, 300, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask2 = cv2.inRange(ret_img, (50, -128, -128), (100, 128, 128))
    YCrCb = cv2.cvtColor(rot_blur, cv2.COLOR_BGR2YCrCb)
    # ret,thresh1 = cv2.threshold(ret_img,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    x, y, w, h = cv2.boundingRect(mask2)
    # cv2.rectangle(rotated,(x,y),(x+w,y+h),(200,0,0),2)
    cropped_image = rotated[y:y + h, x:x + w]

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # threshold grayscale image to extract glare
    mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]

    # Optionally add some morphology close and open, if desired
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # use mask with input to do inpainting
    result = cv2.inpaint(cropped_image, mask, 21, cv2.INPAINT_TELEA)
    proc_bands = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(proc_bands)

    ret, thresh_L = cv2.threshold(l, 65, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask2 = cv2.inRange(proc_bands, (50, -128, -128), (80, 128, 128))
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hsv = cv2.merge([h, s, v])
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    e = v.mean()
    f = s
    v[:, :] = v + round((v.mean() * 0.3))
    s[:, :] = s + round((s.mean() * 0.5))
    hsv = cv2.merge([h, s, v])
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Stacking the original image with the enhanced image

    # colors, count = np.unique(cropped_image.reshape(-1, cropped_image.shape[-1]), axis=0, return_counts=True)
    # print(colors[count.argmax()])
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s += 60
    hsv = cv2.merge([h, s, v])
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    alpha = 3  # Contrast control
    beta = 1.5  # Brightness control

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(hsv, alpha=alpha, beta=beta)

    lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask2 = cv2.inRange(l, 220, 255)
    ret,thresh_L = cv2.threshold(l,70,200,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(mask2,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.imshow('Canny Edges After Contouring', mask2)
    cv2.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(cropped_image, contours, -1, (0, 255, 0), 3)

    cv2.imshow('fEeeE', cropped_image)

    cv2.waitKey(0)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
