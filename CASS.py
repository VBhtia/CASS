import imutils
import time
from firebase import firebase
import json
import urllib
import cv2
import numpy as np
import requests
from urlparse import urlparse

  
#Firebase URL for Consumers 
url2='*********************************'
url1='*********************************' 


#IP Camera 
url='**********************************'

#Mean model values from Caffe model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
a_col=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
g_col = ['Male', 'Female']
 
def initialize_system():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
                        "deploy_age.prototxt", 
                        "age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
                        "deploy_gender.prototxt", 
                        "gender_net.caffemodel")
 
    return (age_net, gender_net)


 
def Phy_Attributes(a_consumer, g_consumer): 

    font = cv2.FONT_HERSHEY_SIMPLEX
    counter=0
    while True:
        imgResp=urllib.urlopen(url)
        
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img=cv2.imdecode(imgNp,-1)

          
        face_cascade = cv2.CascadeClassifier('************/filepath***********')
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        
 
        
        for (a,b,w,h) in faces:
            cv2.rectangle(img,(a,b),(a+w,b+h),(255,255,0),2)
            face_img = img[b:b+h, a:a+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            
           #Predicting the consumer attributes and updating on firebase
            g_consumer.setInput(blob)
            Consumer_g = g_consumer.forward()
           
            C_G = g_col[Consumer_g[0].argmax()]

            a_consumer.setInput(blob)
            Consumer_a = a_consumer.forward()
        
            C_A = a_col[Consumer_a[0].argmax()]
            on_screen = "%s, %s" % (C_G, C_A) 
            if len(faces)>0:
                content='Neutral'
                Users ={'****':'****', '****':'****', '****':'****'}
                data ={'Age Group':C_A, 'Gender':C_G, 'Satisfaction':content}
                Consumer= 'Consumer'+str(counter)
  
                result= requests.put(url2+'/{}.json'.format(Consumer),   data=json.dumps(data))
                result= requests.put(url1+ '/Users.json', data=json.dumps(Users))
                counter=counter+1
            cv2.putText(img, on_screen ,(a,b), font, 1,(255,255,255),2,cv2.LINE_AA)
             
        cv2.imshow("img", img)
 
        key = cv2.waitKey(1) & 0xFF
     
       
        if key == ord("q"):
            break




if __name__ == '__main__':
   a_consumer, g_consumer = initialize_system()
   Phy_Attributes(a_consumer, g_consumer)
   

