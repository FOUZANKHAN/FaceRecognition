import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataset'

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        #cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        if(nbr_predicted==1):
             nbr_predicted='Fouzan'
        elif(nbr_predicted==2):
             nbr_predicted=''
        elif(nbr_predicted==3):
             nbr_predicted='Sohail'
        elif(nbr_predicted==4):
             nbr_predicted='Zubaidi'
        

        cv2.rectangle(im, (x-20,y-80), (x+w+22, y-22), (255,255,255), -1)
        cv2.putText(im, str(nbr_predicted), (x,y-40), font, 1, (0,0,0), 1)       
        #cv2.cv.PutText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
        cv2.imshow('im',im)
        cv2.waitKey(1);


    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 
        
     
        
		


# cleanup the camera and close any open windows
cam.release()

cv2.destroyAllWindows()








