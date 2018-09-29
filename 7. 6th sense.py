import cv2
import imutils
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
prev_stage = 0
counter = 0
while(1):
    #taking each frame
    _, frame = cap.read()
    image = frame.copy()

    #Convert BGR TO HSV
    blurred = cv2.GaussianBlur(image, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #define range of blue color in HSV
    #lower_green = np.array([39,103,48])
    #upper_green = np.array([95,255,255])
    #lower_green = np.array([126,85,55])
    #upper_green = np.array([179,255,255])
    lower_green = np.array([42,50,97])
    upper_green = np.array([95,255,255])

    #Threshold image to only blue color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5,5), np.uint8)
    
    img_erosion = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(img_erosion, kernel, iterations=20)

    # find contours in the thresholded image
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    flag = 0
    tag = 0
    X = [0,0,0,0]
    Y = [0,0,0,0]

    dX = [0,0]
    dY = [0,0]
    
    # loop over the contours
    if 1:
        for c in cnts:
                # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if len(cnts) == 4:
                if flag <= 3:
                    X[flag] = cX
                    Y[flag] = cY
                    flag = flag + 1
                        
            if len(cnts) == 2:
                if tag <= 1:
                    dX[tag] = cX
                    dY[tag] = cY
                    tag = tag + 1

            else:
                flag = 0
                tag = 0
                        
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)

        if len(cnts) == 2:
            for i in range(len(cnts)):
                for j in range(len(cnts)):
                    if (dX[i] < dX[j]):
                        temX = dX[i]
                        temY = dY[i]
                        dX[i] = dX[j]
                        dY[i] = dY[j]
                        dX[j] = temX
                        dY[j] = temY

            cv2.rectangle(image, (dX[0], dY[0]), (dX[1], dY[1]), (0,0,255), 4)
            if counter == 50:
                roi = image[dY[0]:dY[1], dX[0]:dX[1]]
                cv2.imshow('image', roi)
                #cv2.imwrite('Click.png', image)
                #image = cv2.imread('black.jpg',0)
                cv2.imshow('image', image)
                counter = 0
            else:
                counter+=1
                
        elif len(cnts) == 4:
            for i in range(len(cnts)):
                for j in range(len(cnts)):
                    if (X[i] < X[j]):
                        tempX = X[i]
                        tempY = Y[i]
                        X[i] = X[j]
                        Y[i] = Y[j]
                        X[j] = tempX
                        Y[j] = tempY
            if Y[0]<Y[1]:
                tempX = X[0]
                tempY = Y[0]
                X[0]=X[1]
                Y[0]=Y[1]
                X[1]=tempX
                Y[1]=tempY
            if Y[2]<Y[3]:
                tempX = X[2]
                tempY = Y[2]
                X[2]=X[3]
                Y[2]=Y[3]
                X[3]=tempX
                Y[3]=tempY

            print X
            print Y
            cv2.rectangle(image, (X[1], Y[0]), (X[2], Y[3]), (0,0,255), 2)
            cv2.imshow('image', image)  

        #cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
        cv2.imshow('image', image)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        cv2.destroyWindow(image)

cv2.destroyAllWindows()
