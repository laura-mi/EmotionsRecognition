import cv2
from keras.models import load_model
import numpy as np
import sys
from Services.DatasetService import getLabels
from Services.FaceDetectionService import FaceService
from Services.ImageProcessingService import drawDetails
from Services.EmotionService import EmotionService
from Services.GenderService import GenderService

#Get services
emotionService = EmotionService()
genderService = GenderService()
faceService = FaceService()

#Get labels
emotion_labels, gender_labels = getLabels()

#Define parameters 
frame_window = 16
esc_key = 27

#Faces in one window
gender_window = []
emotion_window = []

def main(argv):
	# Video capture
	cv2.namedWindow('window_frame')
	video_capture = cv2.VideoCapture(0)
	k = 0
	#Record until esc button is pressed
	while cv2.waitKey(1) != esc_key:
		input_image = video_capture.read()[1]
		gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
		rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
		
		getEmotionAndGender(gray_image,rgb_image)		
			
		input_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		cv2.imshow('window_frame', input_image)

def getEmotionAndGender(gray_image,rgb_image):
	#Detect faces
	faces = faceService.getFaces(gray_image)
	for coordinates in faces:
		#Detect emotions
		emotion_label_arg = emotionService.getEmotion(coordinates,gray_image)
		emotion_text = emotion_labels[emotion_label_arg]

		#Detect gender
		gender_label_arg = genderService.getGender(coordinates,rgb_image)
		gender_text = gender_labels[gender_label_arg]
		
		#Add emotion to window view
		emotion_window.append(emotion_text)
		gender_window.append(gender_text)

		if len(gender_window) > frame_window:
			emotion_window.pop(0)
			gender_window.pop(0)
		
		#Draw in window
		drawDetails(coordinates,rgb_image,emotion_text,gender_text)
		
if __name__ == "__main__":
   main(sys.argv[1:])
   