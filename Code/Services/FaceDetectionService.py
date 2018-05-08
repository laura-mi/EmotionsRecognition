import cv2
import numpy as np

class FaceService:
	def __init__(self):
		path = '../TrainedModels/haarcascade_frontalface_default.xml'
		self.model = cv2.CascadeClassifier(path)
		
	def getFaces(self,grayImage):
		detection = self.model
		return detection.detectMultiScale(grayImage, 1.3, 5)