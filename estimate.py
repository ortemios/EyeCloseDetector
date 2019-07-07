from keras.models import load_model
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import time

root = os.path.dirname(os.path.abspath(__file__))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(root + "\\assets\\face_landmarks.dat")
cap = cv2.VideoCapture(0)
FOLDER = "_good"
PADDING = 0.4
EYE_RATIO = 0.4
model = load_model(root + "\\model{}\\model.h5".format(FOLDER))#_good
model.load_weights(root + "\\model{}\\weights.hdf5".format(FOLDER))

eyes = [0, 0]

eye_size =  64
	
i = 0
def process_eye_colors(eye):
	eye = cv2.resize(eye, (eye_size, eye_size))
	global i
	cv2.imshow("Eye"+str(i), eye)
	if i == 1:
		i = 0
	else:
		i = 1

	eye = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
	eye = np.reshape(eye, (1, eye.shape[0], eye.shape[1], 3))
	eye = eye/255
	
	out = model.predict(eye)[0][0]
	return out
	
def process_eye_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	
	ratio = (A + B) / (2.0 * C)
	out = abs(ratio - EYE_RATIO) / EYE_RATIO
	
	return 1-out

	
def get_eye_state(frame, shape, start, end, eye_idx):
	eye_pts = []
	for x,y in shape[start:end]:
		eye_pts.append((x,y))
	(x, y, w, h) = cv2.boundingRect(np.array(eye_pts))
	
	d = abs(w - h)
	if w > h:
		y -= d/2
		h += d
	elif h > w:
		x -= d/2
		w += d
	padding = int(w*PADDING)
	x -= padding
	y -= padding
	w += padding*2
	h += padding*2
	x = max(x, 0)
	y = max(y, 0)
		
	eye = frame[int(y):int(y+h), int(x):int(x+w)]
	
	out1 = process_eye_colors(eye)
	out2 = process_eye_ratio(shape[start:end])
	out = (out1+out2)/2
	
	eyes[eye_idx] = out
	
prev_eyes = [0, 0]
state = [1, 1]
	
def parse_frame(frame):
	global file_index
	global prev_eyesW
	
	frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	rects = detector(frame_grey, 1)
	if(len(rects) == 0):
		return frame
	rect = rects[0]
	shape = predictor(frame_grey, rect)
	shape = face_utils.shape_to_np(shape)
	
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	get_eye_state(frame, shape, lStart, lEnd, 0)
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	get_eye_state(frame, shape, lStart, lEnd, 1)
	os.system('cls')
	
	for i in range(0, 2):
		if abs(eyes[i] - prev_eyes[i]) > 0.4:
			state[i] *= -1
		prev_eyes[i] = eyes[i]
	print("{:.10f} {:.10f}".format(eyes[0] > 0.5, eyes[1] > 0.5
	))
	prev_eyes[0] = eyes[0]
	prev_eyes[1] = eyes[1]
	
	key = cv2.waitKey(1)
	if(key == 27):
		exit()
		
	return frame
	
while(True):
	ret, frame = cap.read()
	
	frame = parse_frame(frame)

cap.release()
cv2.destroyAllWindows()
