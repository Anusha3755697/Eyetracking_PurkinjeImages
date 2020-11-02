import cv2
import numpy as np
import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import csv
import pandas as pd
import datetime
 
cap = cv2.VideoCapture("C:\\Users\\manju\\Downloads\\001\\001\\00001.avi")
out = cv2.VideoWriter("C:\\Users\\manju\\OneDrive\\Pictures\\Lab4_screenshots\\output.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500,450))


 
#To read and write Video
#cap = cv2.VideoCapture(gazefile)
#out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500,450))
 
#To check the resolution of Video
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
res = (height,width)
print (res)
#fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)
 
 
#whole condition for tracking video"C:\\Users\\Envy\\Desktop\\research project\\001\\00001.avi"
while cap.isOpened():
    now = datetime.datetime.now()
    ret, frame = cap.read()
    frameRate = 500
    #fgmask = fgbg.apply(frame)
    if ret:
        roi = frame[100:1000, 150:1000]
        rot = cv2.flip(roi,flipCode= -1)
        datet = str(datetime.datetime.now())
        rows, cols, _ = rot.shape
        center_image = (int(rows/2),int(cols/2))
        #cv2.circle(rot, (int(rows/2),int(cols/2)), 5 , (0, 0, 255), 2)
        kernel = np.ones((2,2),np.uint8)
        gray_roi = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (3,3), 0)
        gray_roi = cv2.medianBlur(gray_roi,3)
        gray_roi = cv2.morphologyEx(gray_roi, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.threshold(gray_roi,54, 255, cv2.THRESH_BINARY)[1]
        inverted = cv2.bitwise_not(threshold)
        kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE,ksize = (5,5))
        closing = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel) 
 
        
        threshold = cv2.dilate(threshold, None, iterations=2)
        #threshold = cv2.erode(threshold, None, iterations=1)
        #-----------masking the gray video frame-------------------#
        gray2 = gray_roi.copy()
        mask = np.zeros(gray_roi.shape,np.uint8)
        #------------------Cannyedge detection----------------#
        #edges = cv2.Canny(threshold, 100, 200)
        # Find circles
        # Find circles with HoughCircles
        contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnts = contours[0]
 
        momlist = cv2.moments(cnts)
        if momlist["m00"] != 0:
            dX = int(momlist["m10"] / momlist["m00"])
            dY = int(momlist["m01"] / momlist["m00"]) 
        else:
            dX,dY = 0, 0
        Pupil_cord = (dX,dY)   
        circles = cv2.circle(rot, (dX, dY), 5, (255, 255, 255), -1)
        cv2.putText(rot, "centroid", (dX - 25, dY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
        
        circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, 1, minDist=100000, param1=100, param2=24, minRadius=6,maxRadius=10)
 
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(circles)
            for (x,y,r) in circles:
                cv2.circle(gray_roi, (x,y), r, (0,0,255), 3)
        
        #res = cv2.matchTemplate(gray_roi, template, cv2.TM_CCORR)
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #top_left = min_loc 
        #bottom_right = (top_left[0] + width, top_left[1] + height)
        #cv2.rectangle(gray_roi, top_left, bottom_right, (255, 0, 0), 2)
        #contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        #contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        #cnts = contours[0]
 
        #if 3000<cv2.contourArea(cnts)<100000:
            #(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray_roi)
            #cv2.circle(threshold, maxLoc, 10, (255, 0, 0), 2)
            #Purkinje_img1= maxLoc
            #print (Purkinje_img1)
            #cv2.putText(gray_roi, "c", (maxLoc),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255 ,255),3)
 
        
        #cv2.imshow("te",template)
        cv2.imshow("Frame", threshold)
        cv2.imshow("gray",gray_roi)
 
        
        
        
        #------------------Displaying image/video---------------------------#
        #cv2.imshow("roi", rot)
        #cv2.imshow('Threshold', threshold)
        #cv2.imshow('gray_roi', gray_roi)
        #cv2.imshow("inverted", inverted)
        #cv2.imshow('IMG',closing)
        #cv2.imshow('fgmask',purkinjee)
        #cv2.imshow('frame',result)
        #out.write(roi)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
 
    if cv2.waitKey(15) & 0xFF == ord('q'): # Press 'Q' on the keyboard to exit the playback
        break
cap.release()
#out.release()
cv2.destroyAllWindows()

 