import os
import streamlit as st
from streamlit_router import StreamlitRouter
from streamlit_option_menu import option_menu
from supabase import create_client, Client
import streamlit as st
import hydralit_components as hc
import datetime
import cv2
import csv
import tempfile
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
from FaceDetection import FaceDetector
import datetime,time
import pandas as pd
from st_aggrid import AgGrid,GridOptionsBuilder
import seaborn as sns

url = "https://fzpvefiyomazymkowund.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ6cHZlZml5b21henlta293dW5kIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODI4NTYyNjQsImV4cCI6MTk5ODQzMjI2NH0.qxmzEFCys_1i1Auln9xO1GRMpMYlznSo-qNjbLYK6cI"
supabase = create_client(url, key)

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
drawing_spec = DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=0)
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)




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


	locs = []

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


facenet = cv2.dnn.readNet("model/deploy.prototxt","model/res10_300x300_ssd_iter_140000.caffemodel")

try:
    user = supabase.auth.get_user()
except:
    user = None
page = None
def index(router):
    
    if user is not None:
        user_email = user.user.dict()['email']
        user_data = supabase.table('users').select("*").eq("email",user_email).execute()
        if len(user_data.data) == 0:
            supabase.auth.sign_out()
            router.redirect(*router.build("index"))

        if user_data.data[0]['role'] == 0:
            with st.sidebar:
                page = option_menu(None, ["Home", "Detect From Video","Detect From Webcam", "Logout"],
                                   icons=['house',"film","webcam-fill","box-arrow-right"],key="menu")
        elif user_data.data[0]['role'] == 1:
            with st.sidebar:
                page = option_menu(None, ["Admin Panel","Manage Users", "Logout"],icons=["speedometer2","people-fill","box-arrow-right"],key="menu")
        
    else:
        with st.sidebar:
            page = option_menu(None, ["Home", "Login", "Signup"],icons=["house","door-open","person-plus-fill"],key="menu")


    if page == "Home":
        st.write("# Human Behaviour Detection")
        st.write("""
        
        #### The Human Behaviour Recognition System is an application that uses deep learning classifier models to detect and classify human facial expressions in real-time

        """)

        st.video("video\Media1.mp4",start_time=0)
    elif page == "Login":
        router.redirect(*router.build("login"))

    elif page == "Signup":
        router.redirect(*router.build("signup"))

    elif page == "Detect From Video":
        router.redirect(*router.build("detect_from_video"))

    elif page == "Logout":
        supabase.auth.sign_out()
        router.redirect(*router.build("index"))
    
    elif page == "Detect From Webcam":
        router.redirect(*router.build("detect_from_webcam"))

    elif page == "Manage Users":
        router.redirect(*router.build("manage_users"))

    elif page == "Admin Panel":
        st.write("# Admin Panel")
        st.write(f"""
            #### Welcome {user_data.data[0]['name'].capitalize()} !!
        """)
        users = supabase.table("users").select("*").eq('role',0).execute()
        df= pd.DataFrame(users.data)
        df['date_joined'] = pd.to_datetime(df['created_at']).dt.date
        st.area_chart(df['date_joined'].value_counts())






def signup(router):
    with st.form("Signup"):
        st.write("# Signup")
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Signup")

        if submit_button:
            try:
                res = supabase.auth.sign_up({
                    "email": email,
                    "password": password,
                    })
                user = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password,
                    })
                supabase.table('users').insert({"name": name,"email":email}).execute()
                st.success("Signup successful")
                time.sleep(1)
                router.redirect(*router.build("index"))
            except Exception as e:
                st.error("Signup failed")
                st.write(e)
    if st.button("Back to Homepage"):
        router.redirect(*router.build("index"))


def login(router):
    
    with st.form("Login"):
        st.write("# Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:
            try:
                supabase.auth.sign_in_with_password({"email": email, "password": password})        
                user = supabase.table('users').select("*").eq("email",email).execute()                                     
                if len(user.data) ==0:
                    raise Exception("User not found")
                router.redirect(*router.build("index"))
            except Exception as e: 
                st.error(e.__str__())
                
    if st.button("Back to Homepage"):
        router.redirect(*router.build("index"))

def show_pie_chart():

    try:
        with open("results/data.csv", 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            df = pd.DataFrame(data, columns = ['Emotion', 'data/time'])
            df = df['Emotion']
            count = Counter(df)
            fig, ax = plt.subplots()
            ax.pie(count.values(), labels=count.keys(), autopct='%1.1f%%', shadow=True, startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        os.remove("results/data.csv")
    except:
        pass



        

def detect_from_video(router):
    global emotions_detected
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    with st.sidebar:
        st.subheader("Settings")
        video_file = st.file_uploader("Upload a video",type=["mp4","avi"])
        st.sidebar.markdown("## Settings")
        show_face_mesh = st.sidebar.checkbox("Show Face Mesh",True)

        st.divider()
        if st.button("Go to Homepage"):
            router.redirect(*router.build("index"))


    if video_file is not None:
        tffile = tempfile.NamedTemporaryFile(delete=False)
        tffile.write(video_file.read())
        cap = cv2.VideoCapture(tffile.name)
        emotions_detected = []
        st.write("## Emotion Detection From Video")
        stframe = st.empty()
        stop_detection = False

        col = st.columns(1)
        col[0] = st.empty()
        with col[0]:
            if st.button("STOP"):
                stop_detection = True                
                col[0].empty()

        
        ret=True
        while stop_detection != True and ret==True:
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
                emotions_detected.append(emotion_labels[prediction])
                cv2.putText(frame,f"{emotion_labels[prediction]}",(x,y-10),0,0.5,(255,255,255),2)

                with open("results/data.csv","a", newline="\n") as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow([emotion_labels[prediction], datetime.datetime.now()])

            if show_face_mesh:
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_spec )                    
            
            
            stframe.image(frame,channels="BGR")

        show_pie_chart()

def detect_from_webcam(router):
    
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

    with st.sidebar:
        st.subheader("Settings")
        st.sidebar.markdown("## Settings")
        show_face_mesh = st.sidebar.checkbox("Show Face Mesh",True)
        

        st.divider()
        if st.button("Go to Homepage"):
            router.redirect(*router.build("index"))

    st.write("## Emotion Detection From Webcam")
    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        emotions_detected = []
        stframe = st.empty()
        stop_detection = False

        ret,frame = cap.read()

        if ret == False:
            st.error("Can't Access Webcam")
            st.stop()

        
        col = st.columns(1)
        col[0] = st.empty()
        with col[0]:
            if st.button("STOP"):
                stop_detection = True
                
                col[0].empty()


        while stop_detection != True:
            
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
                emotions_detected.append(emotion_labels[prediction])
                cv2.putText(frame,f"{emotion_labels[prediction]}",(x,y-10),0,0.5,(255,255,255),2)

                with open("results/data.csv","+a", newline="\n") as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow([emotion_labels[prediction], datetime.datetime.now()])

            if show_face_mesh:
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_spec )                    

            ret,frame = cap.read()
            
            
            stframe.image(frame,channels="BGR")
        
        show_pie_chart()
    

            
def manage_users(router):
    st.write("## Manage Users")
    
    users = supabase.table("users").select("*").eq('role',0).execute()
    df= pd.DataFrame(users.data)
    cols = st.columns(4)
    cols[0].write("Name")
    cols[1].write("Email")
    cols[2].write("Action")

    notification = st.empty()
    for user in users.data:
        cols = st.columns(4)
        cols[0].write(f"{user['name']}")
        cols[1].write(f"{user['email']}")
        cols[2].button("Delete",key=f"del{user['id']}", on_click=delete_user, args=(user['id'],notification))
        cols[3].button("Make Admin",key=f"adm{user['id']}", on_click=make_admin, args=(user['id'],notification))

    if st.button("Go to Homepage"):
        supabase.auth.sign_out()
        router.redirect(*router.build("index"))

def make_admin(userid,notification):
    try:
        supabase.table("users").update({"role":1}).eq('id',userid).execute()
        with notification:
            st.success("User made admin successfully")
        time.sleep(1)
        st.experimental_rerun()
    except:
        pass

def delete_user(userid,notification):
    try:
        supabase.table("users").delete().eq('id',userid).execute()
        with notification:
            st.success("User deleted successfully")
        time.sleep(1)
        st.experimental_rerun()
    except:
        pass


    

router = StreamlitRouter()
router.register(index, '/')
router.register(login, '/login')
router.register(signup, '/signup')
router.register(detect_from_video, '/detect-from-video')
router.register(detect_from_webcam, '/detect-from-webcam')
router.register(manage_users, '/manage-users')

router.serve()