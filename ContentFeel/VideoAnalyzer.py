import cv2
import glob
import random
import numpy as np
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
from functions import crop_faces

fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier

def load_model(filepath):
    try :
        fishface.read(filepath)
        print("Model successfully loaded")
    except :
        print("Unable to load model from xml file. Please check your files and try again.")
        input("Press Enter to continue")
        exit()

load_model("googleCKPlus.xml") #load existing model

video = cv2.VideoCapture("videos/sadMarkiplier.avi")

# Check if camera opened successfully
if (video.isOpened()== False): 
  print("Error opening video stream or file")
 
raw_frames = []
cropped = []
rectangles = []
size = []

#4 different classification methods
faceDet = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")
i = 1
# Read until video is completed
while(video.isOpened()):
  # videoture frame-by-frame
  print("processing frame")
  ret, frame = video.read()
  if ret == True:
    height, width, layers = frame.shape
    size = (width,height)
    #raw_frames.append(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
        #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        gray = gray[y:y+h, x:x+w] #Cut the frame to size
        try:
            print("trying to resize frame", i)
            out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            #cropped.append(out) #add to array
            rec = cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2) #draw rectangle around face
            #rectangles.append(rec) #save rectangle image
        except:
            print("Not a valid file")
            pass #If error, pass file
    
    #PREDICTIONS
    pred, conf = fishface.predict(out)
        #write on img
    info1 = 'Guessed emotion: ' + emotions[pred]
    cv2.putText(rec,info1, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,100,0))
    rectangles.append(rec)
    i = i + 1 
  # Break the loop
  else: 
    break

out = cv2.VideoWriter('Analyzed_Videos/analyzedSadMarkiplier.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for img in rectangles:
    #write to video
    out.write(img)
out.release()