from FaceDetection import FaceDetector
from tensorflow.keras.models import load_model
import cv2
import numpy as np

fd = FaceDetector()

cap = cv2.VideoCapture("test/video.mp4")
model = load_model("model/model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
while True:
    ret,frame = cap.read()
    if not ret:
        break
    faces = fd.detect(frame)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        roi = frame[y:y+h,x:x+w]
        roi = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi,(48,48),interpolation=cv2.INTER_AREA)
        roi = np.array(roi)
        roi = roi/255.0
        roi = np.expand_dims(roi, axis=0)
        prediction = np.argmax(model.predict(roi))
        cv2.putText(frame,f"{emotion_labels[prediction]}",(x,y-10),0,0.5,(255,255,255),1)


    cv2.imshow("Video",frame)


    key = cv2.waitKey(1)
    if key == 27 :
        break