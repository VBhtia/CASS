# Import necessary libraries
import imutils
import time
from firebase import firebase
import json
import urllib
import cv2
import numpy as np
import requests
from urlparse import urlparse

# Firebase URLs for Consumers
url2='*********************************'
url1='*********************************' 

# IP Camera URL
url='**********************************'

# Mean model values from Caffe model (used for preprocessing)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender labels
a_col=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
g_col = ['Male', 'Female']

def initialize_system():
    """
    Load the pre-trained age and gender models from Caffe.
    """
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
        "deploy_age.prototxt", 
        "age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt", 
        "gender_net.caffemodel")
 
    return (age_net, gender_net)

def Phy_Attributes(a_consumer, g_consumer): 
    """
    Capture video stream from IP camera, detect faces, predict age and gender,
    and update the results to Firebase.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    counter = 0
    while True:
        # Read the image from IP camera
        imgResp = urllib.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)

        # Load the face detection model
        face_cascade = cv2.CascadeClassifier('************/filepath***********')
        
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Loop through detected faces
        for (a, b, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (a, b), (a+w, b+h), (255, 255, 0), 2)
            face_img = img[b:b+h, a:a+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            g_consumer.setInput(blob)
            Consumer_g = g_consumer.forward()
            C_G = g_col[Consumer_g[0].argmax()]

            # Predict age
            a_consumer.setInput(blob)
            Consumer_a = a_consumer.forward()
            C_A = a_col[Consumer_a[0].argmax()]

            # Display the results on the screen
            on_screen = "%s, %s" % (C_G, C_A) 
            if len(faces) > 0:
                content = 'Neutral'
                Users = {'****': '****', '****': '****', '****': '****'}
                data = {'Age Group': C_A, 'Gender': C_G, 'Satisfaction': content}
                Consumer = 'Consumer' + str(counter)

                # Update Firebase with the consumer attributes
                result = requests.put(url2 + '/{}.json'.format(Consumer), data=json.dumps(data))
                result = requests.put(url1 + '/Users.json', data=json.dumps(Users))
                counter += 1
            cv2.putText(img, on_screen, (a, b), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
             
        # Display the image with the results
        cv2.imshow("img", img)

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

if __name__ == '__main__':
    # Initialize the system and start attribute detection
    a_consumer, g_consumer = initialize_system()
    Phy_Attributes(a_consumer, g_consumer)
