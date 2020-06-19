import cv2
import numpy as np
import math
import matplotlib
from timeit import default_timer as timer
import csv
import pandas as pd


#To read and write Video
cap = cv2.VideoCapture("C:\\Users\\manju\\Downloads\\012\\012\\0000.avi")
out = cv2.VideoWriter("C:\\Users\\manju\\OneDrive\\Pictures\\Lab4_screenshots\\output.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (500,450))

#To check the resolution of Video
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
res = (height,width)
#print (res)
#fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=False)

 
#whole condition for tracking video
while cap.isOpened():
    ret, frame = cap.read()
    #fgmask = fgbg.apply(frame)
    if ret:
        roi = frame[50:900, 150:700]
        start = timer()
        rows, cols, _ = roi.shape
        center_image = (int(rows/2),int(cols/2))
        cv2.circle(roi, (int(rows/2),int(cols/2)), 5 , (0, 0, 255), 2)
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray_roi, 255, 255, cv2.THRESH_BINARY_INV)
        # Gaussian Blurr is to remove unwanted noise in the frame
        gray_roi = cv2.GaussianBlur(gray_roi, (11, 11), 0)
        gray_roi = cv2.medianBlur(gray_roi, 5)
        _,threshold = cv2.threshold(gray_roi, 75, 255, cv2.THRESH_BINARY_INV)
        
        #threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=4)
        #-----------masking the gray video frame-------------------#
        gray2 = gray_roi.copy()
        mask = np.zeros(gray_roi.shape,np.uint8)
        #------------------Cannyedge detection----------------#
        edges = cv2.Canny(threshold, 100, 200)
        #------------------Find Contours-----------------------# 
        contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnts = contours[0]
          
        for cnt in contours:
            if 200<cv2.contourArea(cnt)<5000:
                cv2.drawContours(roi,[cnt],0,(0,255,0),2)
                cv2.drawContours(mask,[cnt],0,255,-1)
                cv2.bitwise_not(gray2,gray2,mask)
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            radius_pupil = int((h)/3)
            cv2.circle(roi, (x + int(w/2), y + int(h/2)), int((h)/3), (0, 0, 255), 2)
            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (50, 200, 0), 1)
            cv2.line(roi, (0, y + int(h/2)), (cols , y + int(h/2)), (50, 200, 0), 1)
            cv2.putText(roi, text = '[Press Q to Exit]', org = (int(cols - 180), int(rows - 15)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.6, color = (0, 0, 0))
            cv2.putText(threshold, text = '[Press Q to Exit]', org = (int(cols - 180), int(rows - 15)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.6, color = (255, 255, 255))
            cv2.putText(gray_roi, text = '[Press Q to Exit]', org = (int(cols - 180), int(rows - 15)), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.6, color = (0, 0, 0))
            break
        
        #--------------finding center of pupil----------------#
        momlist = cv2.moments(cnts)
        if momlist["m00"] != 0:
            dX = int(momlist["m10"] / momlist["m00"])
            dY = int(momlist["m01"] / momlist["m00"]) 
        else:
            dX,dY = 0, 0
        #print (Pupil_cord) 
        circles = cv2.circle(roi, (dX, dY), 5, (255, 255, 255), -1)
        cv2.putText(roi, "centroid", (dX - 25, dY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 

        #sr = pd.DataFrame([center_image,Pupil_cord])
        #Standard_dev = sr.std(skipna = True)
        #print (Standard_dev)

            
        #def calculateDistance(dX,dY):  
        #    dist = math.sqrt((x)**2 + (y)**2)  
        #   return dist     
        #print  (calculateDistance(dX,dY)) 
        #test = np.reshape(gray2,rows)

        #-----------finding a brightest spot in the image------------------#
        for c in enumerate(contours):
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray2)
            B_spot = cv2.circle(gray2, maxLoc, 10, (0, 0, 255), 2)
            Purkinje_img1 = maxLoc
            #print (B_spot)
            cv2.putText(gray2, ".", (maxLoc),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0 ,255),3)
        endtimer = timer() - start
        print ("totaltime taken = ", endtimer)

        

        #---------------------writing to csv----------------------------#
        with open('C:\\Users\\manju\\OneDrive\\Desktop\\Anusha\\Research project\\Csv_data\\data_Sample_012_006.csv', 'a', newline='') as outfile:   
            fieldnames = ['Pupil_X', 'Pupil_Y','Center_spot']
            output = csv.DictWriter(outfile, fieldnames=fieldnames)
            output.writeheader()
            output.writerow({'Pupil_X':str (dX),'Pupil_Y': str (dY), 'Center_spot': Purkinje_img1})
            output.writerow({})
            outfile.close()
        
        
        #------------------Displaying image/video---------------------------#
        cv2.imshow("roi", roi)
        cv2.imshow('Threshold', threshold)
        cv2.imshow('gray_roi', gray_roi)
        cv2.imshow("Canny", edges)
        cv2.imshow('IMG',gray2)
        #cv2.imshow('fgmask',frame)
        #cv2.imshow('frame',result)
        out.write(roi)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    


    if cv2.waitKey(15) & 0xFF == ord('q'): # Press 'Q' on the keyboard to exit the playback
        break
cap.release()
out.release()
cv2.destroyAllWindows()
