import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64


app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #change_the path_accordingly
ds_factor=0.6

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    faces = detect_faces(image)
    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)
        #for item in faces:
            #draw_rectangle(image, item['rect'])
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)


def detect_faces(img):

    '''Detect face in an image'''

    faces_list = []

    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    if  len(faces) == 0:
        return faces_list
    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {}
        face_dict['face']= gray[y:y+h, x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
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
        face_img = img[y:y+h, h:h+w].copy()#without a copying function your blob object wont work
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
        cv2.putText(img, overlay_text, (x,y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        faces_list.append(face_dict)
    return faces_list

def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=3000)
