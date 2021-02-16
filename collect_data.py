# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:28:20 2021

@author: Lenovo
"""
import cv2
import urllib
import numpy as np

face_data = r"haarcascade_frontalface_default.xml"

#model
classifier = cv2.CascadeClassifier(face_data)

#image url
url = "http://192.168.43.1:8080/shot.jpg"

data = []

while len(data)<100:
    #fetching face images
    image_from_url = urllib.request.urlopen(url)
    frame = np.array(bytearray(image_from_url.read()),np.uint8)
    frame = cv2.imdecode(frame,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    for x,y,w,h in faces:
        face_frame = frame[y:y+h,x:x+w]
        cv2.imshow("only face",face_frame)
        
        if len(data)<100:
            print(len(data)+1,"/100")
            data.append(face_frame)
        else:
            break
    
    cv2.imshow("capture",frame)
    if cv2.waitKey(32)==ord("q"):
        break
    
cv2.destroyAllWindows()

if len(data)==100:
    name = input("Enter Face Doner Name : \n")
    for i in range(0,100):
        cv2.imwrite("images/"+name+"_"+str(i+1)+".jpg",data[i])
    else:
        print("Completed")
else:
    print("Need More Data")
        
    
    
