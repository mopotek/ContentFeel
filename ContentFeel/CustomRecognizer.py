
import cv2
import glob
import random
import numpy as np
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
#fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
data = {}



def save_model(name):
    filename = name + ".xml"
    try:
        fishface.save(filename)
        print("Model successfully saved")
    except:
        print("Could not save model, please check for mistakes")
        input("Press Enter to continue")

def load_model(filepath):
    try :
        fishface.read(filepath)
        print("Model successfully loaded")
    except :
        print("Unable to load model from xml file. Please check your files and try again.")
        input("Press Enter to continue")
        exit()

def update_model(newImages,newLabels):
    fishface = cv2.face.FisherFaceRecognizer_create()
    fishface.train(training_data, np.asarray(training_labels))
    return fishface

def get_files(emotion): #Define function to get file list, randomly shuffle it
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training,prediction 

def prediction_files(emotion):
    files = glob.glob("toPredict\\%s\\*" %emotion)
    random.shuffle(files)
    prediction = files
    return prediction    

def crop_faces(files):
    print("Cropping faces...")
    cropped = []
    filenumber = 0

    #4 different classification methods
    faceDet = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("OpenCV_FaceCascade/haarcascade_frontalface_alt_tree.xml")
    i = 1
    for f in files:
        #frame = cv2.imread(f) #Open image
        #cv2.waitKey(0)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
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
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                print("trying to resize frame", i)
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cropped.append(out) #add to array
            except:
                print("Not a valid file")
                pass #If error, pass file
        filenumber += 1 #Increment image number
        i = i + 1 
    return cropped

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels

def train_model(training_data, training_labels):
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))

def run_recognizer():
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("size of training set is:", len(training_labels), "images")
    #EITHER train & save model OR LOAD MODEL
    #train_model(training_data, training_labels)
    #save_model("googleCKPlus")
    load_model("googleCKPlus.xml")
    #PREDICTIONS
    print("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        if pred == prediction_labels[cnt]:
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
        info1 = 'Guessed emotion: ' + emotions[pred]
        info2 = 'True emotion: ' + emotions[prediction_labels[cnt-1]]
        print(info1,info2)        
        cv2.putText(image,info1, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,100,0))
        cv2.putText(image,info2, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,100,0))
        cv2.namedWindow("Image to recognize")
        cv2.imshow("Image to recognize", image)
        cv2.waitKey(0)
    cv2.destroyWindow("Image to recognize")
    print("got", (100*correct)/(correct + incorrect), "percent correct!")
    return ((100*correct)/(correct + incorrect))

#run_recognizer()
