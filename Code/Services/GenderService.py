import numpy as np
import cv2
from keras.models import load_model

from Services.ImageProcessingService import processForGender
from Services.ImageProcessingService import modifyPosition

class GenderService:
	
	def __init__(self):
		# loading models
		path = '../TrainedModels/genderClassifier.hdf5'
		self.genderClassifier = load_model(path, compile=False)
		
		#reshape
		self.targetSize = self.genderClassifier.input_shape[1:3]
	
	def getGender(self,coordinates,imageInRgb):
		x1, x2, y1, y2 = modifyPosition(coordinates, (10, 10))
		face = imageInRgb[y1:y2, x1:x2]
		
		face = cv2.resize(face, (self.targetSize))		
		face = processForGender(face)
		
		label = np.argmax(self.genderClassifier.predict(face))
		return label
		
