import cv2

class FaceDetector:
    def __init__(self):
        self.classfier = cv2.CascadeClassifier("classcifier/haarcascade_frontalface_default.xml")

    def detect(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return self.classfier.detectMultiScale(gray,1.1,4)


