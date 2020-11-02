import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import csv
import pandas as pd
import datetime

#gazefile = input("Provide filename")
#outputfile = input("Provide file to write")

#To read and write Video
cap = cv2.VideoCapture("C:\\Users\\manju\\Downloads\\001\\001\\00001.avi")
out = cv2.VideoWriter("C:\\Users\\manju\\OneDrive\\Pictures\\Lab4_screenshots\\output.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500,450))

#To read and write Video
#cap = cv2.VideoCapture(gazefile)
#out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500,450))

#To check the resolution of Video
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
res = (height,width)
#print (res)
#fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)

frametime = 200
#whole condition for tracking video"C:\\Users\\Envy\\Desktop\\research project\\001\\00001.avi"
while cap.isOpened():
    now = datetime.datetime.now()
    ret, frame = cap.read()
    #fgmask = fgbg.apply(frame)
    if ret:
        roi = frame[100:1000, 100:1000]
        rot = cv2.flip(roi,flipCode= -1)
        datet = str(datetime.datetime.now())
        rows, cols, _ = rot.shape
        center_image = (int(rows/2),int(cols/2))
        cv2.circle(rot, (int(rows/2),int(cols/2)), 5 , (0, 0, 255), 2)
        kernel = np.ones((2,2),np.uint8)
        gray_roi = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (11,11), 0)
        gray_roi = cv2.medianBlur(gray_roi, 5)
        gray_roi = cv2.morphologyEx(gray_roi, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.threshold(gray_roi,50, 255, cv2.THRESH_BINARY_INV)[1]
        #print ("****")
        #threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)
        #threshold = cv2.erode(threshold, None, iterations=1)
        #-----------masking the gray video frame-------------------#
        gray2 = gray_roi.copy()
        mask = np.zeros(gray_roi.shape,np.uint8)
        #------------------Cannyedge detection----------------#
        #edges = cv2.Canny(threshold, 100, 200)
        #------------------Find Contours-----------------------# 
        contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnts = contours[0]
          
        for cnt in contours:
            if 20000<cv2.contourArea(cnt)<25000:
                cv2.drawContours(rot,[cnt],-1,(0,255,0),3)
                cv2.drawContours(mask,[cnt],-1,255,-1)
                cv2.bitwise_not(gray2,gray2,mask)
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(rot, (x, y), (x + w, y + h), (255, 0, 0), 2)
            radius_pupil = int((h)/3)
            cv2.circle(rot, (x + int(w/2), y + int(h/2)), int((h)/3), (0, 0, 255), 2)
            cv2.line(rot, (x + int(w/2), 0), (x + int(w/2), rows), (50, 200, 0), 1)
            cv2.line(rot, (0, y + int(h/2)), (cols , y + int(h/2)), (50, 200, 0), 1)
            cv2.putText(rot, text = "press q to quit", org = (int(cols - 180), int(rows - 15)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.6, color = (0, 0, 0))
            cv2.putText(threshold,text = "press q to quit",  org = (int(cols - 180), int(rows - 15)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.6, color = (255, 255, 255))
            cv2.putText(gray_roi,text = "press q to quit", org = (int(cols - 180), int(rows - 15)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.6, color = (0, 0, 0))
            break
        
        #--------------finding center of pupil----------------#
        momlist = cv2.moments(cnts)
        if momlist["m00"] != 0:
            dX = int(momlist["m10"] / momlist["m00"])
            dY = int(momlist["m01"] / momlist["m00"]) 
        else:
            dX,dY = 0, 0
        Pupil_cord = (dX,dY) 
        print (Pupil_cord)  
        circles = cv2.circle(rot, (dX, dY), 5, (255, 255, 255), -1)
        cv2.putText(rot, "centroid", (dX - 25, dY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 
        

       
            
        
        #-----------finding a brightest spot in the image------------------#
        #if 10000<cv2.contourArea(cnts)<100000:
        for c in enumerate(contours):
            if 14000<cv2.contourArea(cnts)<18000:
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray_roi)
                p1= cv2.circle(gray_roi, maxLoc, 10, (0, 255, 0), 2)
                Purkinje_img1 = maxLoc
        #     (x, y, w, h) = cv2.boundingRect(c)
	    #     #((cX, cY), radius) = cv2.minEnclosingCircle(c)
            
        #     #t = cv2.circle(gray_roi, (int(cX), int(cY)), int(radius),(200, 200, 0), 2)
        #     cv2.line(gray_roi, (x + int(w/2), 0), (x + int(w/2), rows), (50, 200, 0), 1)
        #     cv2.line(gray_roi, (0, y + int(h/2)), (cols , y + int(h/2)), (50, 200, 0), 1)
        #     #print (Purkinje_img1)
                cv2.putText(gray_roi, ".", (maxLoc),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0 ,255),3)
       
                   
                

           
            
              
	           
        for p in (cnts):
            #if 14000<cv2.contourArea(cnts)<19000:
            mlist = cv2.moments(p)
            if mlist["m00"] != 0:
                p1X = int(mlist["m10"] / mlist["m00"])
                p1Y = int(mlist["m01"] / mlist["m00"]) 
            else:
                p1X,p1Y = 0, 0
            Pur_1 = (p1X,p1Y) 
        #print (Pupil_cord)  
            c = cv2.circle(gray_roi, (p1X, p1Y), 5, (255, 255, 255), -1)
            cv2.putText(gray_roi, "centroid", (p1X - 25, p1Y - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 

        #---------------------writing to csv----------------------------#
        #with open('C:\\Users\\Envy\\Desktop\\subject1purkinjecapture.csv', 'a', newline='') as outfile:   
            #fieldnames = ['date','Pupil_X','Pupil_Y','Purkinje']
            #output = csv.DictWriter(outfile, fieldnames=fieldnames)
            #output.writeheader()
            #output.writerow({'date':now,'Pupil_X':str(dX),'Pupil_Y':str(dY),'Purkinje':Purkinje_img1})
            #output.writerow({})
            #outfile.close()
        
        
        #------------------Displaying image/video---------------------------#
        cv2.imshow("roi", rot)
        cv2.imshow('Threshold', threshold)
        cv2.imshow('gray_roi', gray_roi)
        #cv2.imshow("Canny", edges)
        #cv2.imshow('IMG',gray2)
        #cv2.imshow('fgmask',frame)
        #cv2.imshow('frame',result)
        #out.write(roi)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    


    if cv2.waitKey(15) & 0xFF == ord('q'): # Press 'Q' on the keyboard to exit the playback
        break
cap.release()
#out.release()
cv2.destroyAllWindows()