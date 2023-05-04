import streamlit as st
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
from FaceDetection import FaceDetector
import cv2
import numpy as np
import tempfile
st.title("Human Behaviour Recognition")


    

@st.cache_resource
def load_face_detection():
    return FaceDetector()

fd = load_face_detection()

@st.cache_resource
def load_classifier_model():
    model = load_model("model/model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_classifier_model()

mp_drawing = mp.solutions.drawing_utils
drawing_spec = DrawingSpec(color=(255, 0, 0 , 10), thickness=1, circle_radius=0)
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


APP_MODE = st.sidebar.selectbox("Choose the App Mode", ["Detect From Video","About App", "Detect From Webcam", "detect faces"])

facenet = cv2.dnn.readNet("model/deploy.prototxt","model/res10_300x300_ssd_iter_140000.caffemodel")

@st.cache_data
def detect_face(frame, _faceNet,threshold=0.5):
	# grab the dimensions of the frame and then construct a blob
	# from it
	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	_faceNet.setInput(blob)
	detections = _faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	locs = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
			# add the face and bounding boxes to their respective
			# lists
			locs.append((startX, startY, endX, endY))

	return (locs)



if APP_MODE == "About App":
    st.markdown(''' 
    ## About Human Behavior Recognition System

    Human Behavior Recognition System is an application that uses deep learning techniques to classify emotions from facial expressions in real-time. This system utilizes a Convolutional Neural Network (CNN) algorithm for emotion classification, and OpenCV for video processing. Additionally, the application features a graphical user interface built with Streamlit, which enables users to easily record video and view the system's output.

    ''')
    st.video("./video/Media1.mp4")

elif APP_MODE == "Detect From Video":
    video_file = st.file_uploader("Upload a video",type=["mp4","avi"])
    st.sidebar.markdown("## Settings")
    st.sidebar.checkbox("Show Face Mesh",True)
    st.sidebar.color_picker("Drawing Color","#ffffff")
    if video_file is not None:
        tffile = tempfile.NamedTemporaryFile(delete=False)
        tffile.write(video_file.read())
        cap = cv2.VideoCapture(tffile.name)
        stframe = st.empty()
        while True:
            ret,frame = cap.read()
            
            if not ret:
                break
            faces = detect_face(frame,facenet,0.5)
            for (x,y,x2,y2) in faces:
                cv2.rectangle(frame,(x,y),(x2,y2),(255,255,255),2)
                roi = frame[y:x2,x:y2]
                roi = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi,(48,48),interpolation=cv2.INTER_AREA)
                roi = np.array(roi)
                roi = roi/255.0
                roi = np.expand_dims(roi, axis=0)
                prediction = np.argmax(model.predict(roi))
                cv2.putText(frame,f"{emotion_labels[prediction]}",(x,y-10),0,0.5,(255,255,255),2)

            # frame.flags.writable = True
            # results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # if results.multi_face_landmarks:
            #     for face_landmarks in results.multi_face_landmarks:
            #         mp_drawing.draw_landmarks(
            #             image=frame,
            #             landmark_list=face_landmarks,
            #             connections=mp_face_mesh.FACEMESH_TESSELATION,
            #             landmark_drawing_spec=None,
            #             connection_drawing_spec=drawing_spec )                    

            
            stframe.image(frame,channels="BGR")

elif APP_MODE == "Detect From Webcam":
    startDetecting = False
    indexs = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            indexs.append(i)
        cap.release()
    
    
    if st.button("Start"):
        startDetecting = True
    if startDetecting:
        cap = cv2.VideoCapture(1)
        stframe = st.empty()
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            faces = detect_face(frame,facenet,0.5)
            for (x,y,x2,y2) in faces:
                cv2.rectangle(frame,(x,y),(x2,y2),(255,255,255),2)
                roi = frame[y:x2,x:y2]
                roi = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(roi,(48,48),interpolation=cv2.INTER_AREA)
                roi = np.array(roi)
                roi = roi/255.0
                roi = np.expand_dims(roi, axis=0)
                prediction = np.argmax(model.predict(roi))
                cv2.putText(frame,f"{emotion_labels[prediction]}",(x,y-10),0,0.5,(255,255,255),1)
            stframe.image(frame,channels="BGR")

