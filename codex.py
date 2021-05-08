import face_recognition
import dlib
import cv2 as cv
import os
import numpy as np
from datetime import datetime

# importing images
path = "../prj_1/Class_1A/Faces"
Faces,Names = [],[]
myList=os.listdir(path)
# print(myList)

for cls in myList:
    currentImg=cv.imread(f"{path}/{cls}")
    Faces.append(currentImg)
    Names.append(os.path.splitext(cls)[0]) # removes jpg
# print(Names)

# encoding
def findEncoding(imgList):
    encodedList=[]
    for img in imgList:
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

encodeForKnownFaces = findEncoding(Faces)
print(len(encodeForKnownFaces)," \n! ENCODING COMPLETE !")

web_cam = cv.VideoCapture(0)
ip_address="ip address of the ipCam"
web_cam.open(ip_address)

def markAttandance(name):
    with open("Class_1A/Faces/Attandance.csv","r+") as f:                   # r+ is for right +
        myDataList=f.readlines()
        nameList=[] #all the names we find
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now() # gives us date and time
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")

while True:
    success,webimg = web_cam.read()

    # reduce the size of the images for speeding up the process
    webimg = cv.resize(webimg,(640,480),None,0.25,0.25)
    webimg = cv.cvtColor(webimg,cv.COLOR_BGR2RGB)

    # finding and encoding as there can be multiple faces
    faceLocations = face_recognition.face_locations(webimg)
    encodeCurrentFrame = face_recognition.face_encodings(webimg,faceLocations)

    # finding the matches ; Comparing
    for encodeFace, faceLoc in zip(encodeCurrentFrame,faceLocations):
        matches = face_recognition.compare_faces(encodeForKnownFaces,encodeFace)
        faceDist = face_recognition.face_distance(encodeForKnownFaces,encodeFace) # it will return us a list as well the lowest the best
        # print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name=Names[matchIndex]
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv.rectangle(webimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(webimg, (x1, y2-35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(webimg, name, (x1+6,y2-6), cv.FONT_HERSHEY_COMPLEX, 1, (0,0, 255), 2)

            markAttandance(name)

    cv.imshow('video',webimg)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break