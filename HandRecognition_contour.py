#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np
import math

def capture_histogram(source):
    cap=cv2.VideoCapture(source)
    while True:
        _,frame=cap.read()
        frame=cv2.flip(frame,1)
        frame=cv2.resize(frame,(1000,600))
        
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'Place region of hand in sied the box and press a',(5,50),font,0.7,(255,255,255),2,cv2.LINE_AA)
        #above line is for text display above the box
        cv2.rectangle(frame,(500,100),(580,180),(105,105,105),2)
        box = frame[105:175,505:575]
        
        cv2.imshow("capture Histogram",frame)
        key=cv2.waitKey(10)
        if key == ord('a'):
            object_color = frame
            cv2.destroyAllWindows()
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
    object_color_hsv=cv2.cvtColor(object_color,cv2.COLOR_BGR2HSV)
    object_hist=cv2.calcHist([object_color_hsv],[0,1],None,[12,15],[0,180,0,256])
    cv2.normalize(object_hist,object_hist,0,255,cv2.NORM_MINMAX)
    return object_hist


# In[11]:


def locate_object(frame,object_hist):
    hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #apply back propagation to image using object_hist as
    object_segment=cv2.calcBackProject([hsv_frame],[0,1],object_hist,[0,180,0,256],1)
    _,segment_thresh=cv2.threshold(object_segment,20,255,cv2.THRESH_BINARY)
    #APPLYING SOME IMAGE OPERATIONS TO ENHANCE IMAGE
    kernel=None
    disc=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    filtered=cv2.filter2D(segment_thresh,-1,disc)
    
    eroded=cv2.erode(filtered,kernel,iterations=2)
    dilated=cv2.dilate(eroded,kernel,iterations=2)
    closing=cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel)
    
    #masking
    masked=cv2.bitwise_and(frame,frame,mask=closing)
    
    return closing,masked,segment_thresh

def detect_hand(frame,hist):
    return_value={}
    detect_hand,masked,raw=locate_object(frame,hist)
    return_value["binary"]=detect_hand
    return_value["masked"]=masked
    return_value["raw"]=raw
    
    image,contours=cv2.findContours(detect_hand,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    palm_area=0
    flag=None
    cnt=None
    
    for (i,c) in enumerate(contours):
        area=cv2.contourArea(c)
        if area>palm_area:
            palm_area=area
            flag=i
    if flag is not None and palm_area > 10000:
        cnt=contours[flag]
        return_value["contours"]=cnt# largest contour which we found
        cpy=frame.copy()
        cv2.drawContours(cpy,[cnt],0,(0,255,0),2)
        return_value["boundaries"]=cpy
        return True,return_value
    else:
        return False, return_value
    


# In[12]:


import cv2
import numpy as np
#from HandRecognition import *

cap=cv2.VideoCapture(0)
hist=capture_histogram(0)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    hand_detected,hand=detect_hand(frame,hist)
    if hand_detected:
        hand_image=hand["boundaries"]
        cv2.imshow("Hand Detector",hand_image)
    else:
        cv2.imshow("Hand Detector1",frame)
    #cv2.imshow("Raw",hand["raw"])
    #cv2.imshow("Enhanced Binary",hand["binary"])
    #cv2.imshow("Masked",hand["masked"])
    
    k=cv2.waitKey(10)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[13]:




