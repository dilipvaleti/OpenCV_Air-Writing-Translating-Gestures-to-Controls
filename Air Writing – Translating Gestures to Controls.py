#!/usr/bin/env python
# coding: utf-8

# In[5]:


from HandRecognition_contour import *
import cv2
import numpy as np

cap=cv2.VideoCapture(0)
#initialize a black canvas
screen=np.zeros((600,1000))

hist=capture_histogram(0)
    
curr=None
prev=None

while True:
    ret,frame=cap.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(1000,600))
    
    hand_detected,hand=detect_hand(frame,hist)
    if hand_detected:
        hand_image=hand["boundaries"]
        
        finguretips=extract_finguretips(hand)
        plot(hand_image,finguretips)
        
        prev=curr
        curr=finguretips[0]
        
        if prev and curr:
            cv2.line(screen,prev,curr,(255,0,0),5)
        cv2.imshow("Drawing",screen)
        ccv2.imshow("hand detector", hand_image)
    else:
        cv2.imshow("hand Detector",frame)
        
    k=cv2.waitKeyKey(5)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    

