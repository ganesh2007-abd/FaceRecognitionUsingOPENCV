import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier("IMAGES_TUTS\\FaceDetection\\haar_face_def.xml")
people = ['ViratKohli','RohitSharma','Dhoni']


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trainedmodel.yml")

img = cv.imread(r"C:\Users\HP\Downloads\TestImage9.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_rect = haar_cascade.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in img_rect:
    face_roi = gray[y:y+h,x:x+w]
    label,conf = face_recognizer.predict(face_roi)

    print(f"The pictures contain {people[label]} with confidence {conf}")

    cv.putText(img,people[label],(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow("Result",img)
cv.waitKey(0)