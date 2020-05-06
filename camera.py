import numpy as np
import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #change_the path_accordingly
ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def get_frame(self):
        ret, frame = self.video.read()
        frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roigray = gray[y:y+h, x:x+w]
            roicolor = frame[y:y+h, x:x+w]
            #ret, jpeg = cv2.imencode('.jpg', frame)
            #Age and Gender detection
            faceProto="opencv_face_detector.pbtxt"
            faceModel="opencv_face_detector_uint8.pb"
            ageProto="age_deploy.prototxt"
            ageModel="age_net.caffemodel"
            genderProto="gender_deploy.prototxt"
            genderModel="gender_net.caffemodel"
            MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
            ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            genderList=['Male','Female']
            faceNet=cv2.dnn.readNet(faceModel,faceProto)
            ageNet=cv2.dnn.readNet(ageModel,ageProto)
            genderNet=cv2.dnn.readNet(genderModel,genderProto)
            face_img = frame[y:y+h, h:h+w].copy()#without a copying function your blob object wont work
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            #Predict Gender
            genderNet.setInput(blob)
            gender_preds = genderNet.forward()
            gender = genderList[gender_preds[0].argmax()]
            print("Gender : " + gender)
            #Age_prediction
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("AGE Output : {}".format(agePreds))
            print("Age : {}".format(age))
            overlay_text = "%s %s" % (gender, age)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, overlay_text, (x,y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
