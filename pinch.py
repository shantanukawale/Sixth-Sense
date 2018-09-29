import cv2
import imutils
import time
import numpy as np
import math

img = cv2.imread('mbuntu-1.jpg')
h,w,s = img.shape
a = float(w)/h
cv2.namedWindow('zoom',cv2.WINDOW_NORMAL)
cv2.resizeWindow('zoom',w,h)
cv2.imshow('zoom', img)
i = 0
j = 0

y0 = 0
y1 = h
x0 = 0
x1 = w

z = 20

px = 0
py = 0

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
current_distance = 0
prev_distance = 50
while(1):
    #taking each frame
    _, frame = cap.read()
    image = frame.copy()
    image1 = frame.copy()

    #Convert BGR TO HSV
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #define range of blue color in HSV
    #lower_green = np.array([39,103,48])
    #upper_green = np.array([95,255,255])
    lower_green = np.array([42,50,97])
    upper_green = np.array([95,255,255])

    #Threshold image to only blue color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    
    img_erosion = cv2.erode(mask, kernel, iterations=5)
    mask = cv2.dilate(img_erosion, kernel, iterations=15)

    # find contours in the thresholded image
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    tag = 0
    dX = [0,0]
    dY = [0,0]
    X = [0]
    Y = [0]
    # loop over the contours
    if 1:
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            if len(cnts) == 2:
                if tag <= 1:
                    dX[tag] = cX
                    dY[tag] = cY
                    tag = tag + 1            
            elif len(cnts) == 1:
                X[0] = cX
                Y[0] = cY
                if (X[0]-px)>10:
                    if (x1<w and (x1+a*z)< w):
                        Click = img[int(y0):int(y1), int(x0+a*z): int(x1+a*z)]
                        cv2.imshow('zoom',Click)
                        x0+=a*z
                        x1+=a*z
    
                elif (px-X[0])>10:
                    if (x0>0 and (x0-a*z)> 0):
                        Click = img[int(y0):int(y1), int(x0-a*z): int(x1-a*z)]
                        cv2.imshow('zoom',Click)
                        x0-=a*z
                        x1-=a*z
                elif (py-Y[0])>10:
                    if (y1<h and (y1+z)<h):
                        Click = img[int(y0+z):int(y1+z), int(x0): int(x1)]
                        cv2.imshow('zoom',Click)
                        y0+=z
                        y1+=z
                elif (Y[0]-py)>10:
                    if (y0>0 and (y0-z)>0):
                        Click = img[int(y0-z):int(y1-z), int(x0): int(x1)]
                        cv2.imshow('zoom',Click)
                        y0-=z
                        y1-=z
                
                px = X[0]
                py = Y[0]
                #print X[0],Y[0]
                
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

        if len(cnts) == 2:
            if dX[0]>dX[1]:
                tempX = dX[0]
                tempY = dY[0]
                dX[0]=dX[1]
                dY[0]=dY[1]
                dX[1]=tempX
                dY[1]=tempY
            #print dX,dY
            current_distance = ((dX[1]-dX[0])**2 + (dY[1]-dY[0])**2)**0.5
            if current_distance - prev_distance > 4:
                Click = img[int(y0+z):int(y1-z), int(x0+a*z): int(x1-a*z)]
                cv2.imshow('zoom',Click)
                y0+=z
                y1-=z
                x0+=a*z
                x1-=a*z
                
            elif prev_distance -current_distance > 4:
                if (y0-z>0 and y1+z<h and (x0-a*z)>0 and (x1+a*z)<w):
                    Click = img[int(y0-z):int(y1+z), int(x0 - a*z): int(x1+a*z)]
                    cv2.imshow('zoom',Click)
                    y0-=z
                    y1+=z
                    x0-=a*z
                    x1+=a*z
                
            prev_distance = current_distance
            #print current_distance


        #cv2.imshow('image',image)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

cv2.destroyAllWindows()
