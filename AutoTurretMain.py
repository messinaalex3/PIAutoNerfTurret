import cv2
import numpy as np
import argparse
import mediapipe as mp
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def detect(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    centerPoint = frame.shape[1]//2
    boxes = []
    faces = detector.detectMultiScale(frame,scaleFactor=1.05,
			minNeighbors=9, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)
    for face in faces:
        cv2.rectangle(frame,(face[0],face[1]), (face[0] + face[2], face[1] + face[3]),(255,0,0),2)
        distFromCenter = centerPoint - (face[0] + face[2]/2)
        if abs(distFromCenter) < frame.shape[1]//2 * .2:
            print("FIRE!!!!")
        elif distFromCenter > 0:
            print("turn right!")
        else:
            print("turn Left")
    
    cv2.imshow("capture",frame)

def detectMediaPipe(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = face_detection.process(frame)
    if faces.detections:
        for face in faces.detections:
            center = mp_face_detection.get_key_point(
                    face, mp_face_detection.FaceKeyPoint.NOSE_TIP).x
            if center < .6 and center > .4:
                print("fire")
            elif center < .4:
                print("turn left")
            else:
                print("turn right")

cam = cv2.VideoCapture(0)
detector= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
face_detection = mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=.6)
while True:
    start_time = time.time()
    ret,frame = cam.read()
    detectMediaPipe(frame)
    #detect(frame)
    print("FPS: ",1.0/(time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()