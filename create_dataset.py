from imutils import face_utils
from os.path import *
from os import listdir
from os import makedirs
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time

root = dirname(abspath(__file__)) + "\\"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(root + "\\assets\\face_landmarks.dat")
file_index = int(time.time())

categories = ["open","close"]
batch_ratio = 1

def extract_region(frame, shape, start, end):
	points = np.array(shape[start:end])
	(x, y, w, h) = cv2.boundingRect(points)
	
	d = abs(w - h)
	if w > h:
		y -= d/2
		h += d
	elif h > w:
		x -= d/2
		w += d
		
	padding = int(w*0.4)
	x -= padding
	y -= padding
	w += padding*2
	h += padding*2
	x = max(x, 0)
	y = max(y, 0)
	w = min(w, frame.shape[1]-x)
	h = min(h, frame.shape[0]-y)
		
	out =  frame[int(y):int(y+h), int(x):int(x+w)]
	
	return out

def process_sample(filename, category, file_counter):
	global file_index

	frame = cv2.imread(filename)
	
	frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(frame_grey, 1)
	
	for rect in rects:
		batch = "train"
		if file_counter == batch_ratio:
			file_counter = 0
			batch = "validation"
		else:
			file_counter += 1
			
		sample_dir = root + "dataset\\{}\\{}".format(batch, category)
		try:
			makedirs(sample_dir)
		except Exception as e:
			pass
				
		shape = predictor(frame_grey, rect)
		shape = face_utils.shape_to_np(shape)
		
		eye = extract_region(frame, shape, 36, 41)
		cv2.imwrite(sample_dir + "\\{}.left.jpg".format(file_index), eye)
		eye = extract_region(frame, shape, 42, 47)
		cv2.imwrite(sample_dir + "\\{}.right.jpg".format(file_index), eye)
	
		file_index += 1
		
	return file_counter
	
def main():

	for category in categories:
		file_counter = 0
		samples_dir = root + "samples\\{}\\".format(category)
		for file in listdir(samples_dir):
			filename = join(samples_dir, file)
			if not isfile(filename):
				continue
				
			try:
				print(filename)
				file_counter = process_sample(filename, category, file_counter)
			except Exception as e:
				print(e)
	
main()
