import cv2 as cv
import os
import numpy as np

haar_cascade = cv.CascadeClassifier('IMAGES_TUTS\\FaceDetection\\haar_face_def.xml')

DIR = r'D:\SKILLS\OPENCV\IMAGES_TUTS\FaceRecognitionImages'
people = ['ViratKohli','RohitSharma','Dhoni']

labels = []
features = []

def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for image in os.listdir(path):
            img_path = os.path.join(path,image)
            col = cv.imread(img_path)
            img = cv.cvtColor(col,cv.COLOR_BGR2GRAY)
            if img is None:
                continue
            img_rect = haar_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)

            if len(img_rect) == 0:
                print(f"No image found in : {img_path}")
            if len(img_rect) > 1:
                print(f"More than 1 image found in : {img_path}")
            for (x,y,w,h) in img_rect:
                img_roi = img[y:y+h,x:x+w]
                features.append(img_roi)
                labels.append(label)


create_train()
features = np.array(features,dtype='object')
labels = np.array(labels)

facerecognizer = cv.face.LBPHFaceRecognizer_create()
facerecognizer.train(features,labels)

print(len(features))
print(len(labels))

print(features)
print(labels)

facerecognizer.save("face_trainedmodel.yml")
np.save('features.npy',features)
np.save('labels.npy',labels)