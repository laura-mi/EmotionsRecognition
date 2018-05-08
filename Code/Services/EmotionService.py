import numpy as np
import cv2
from keras.models import load_model

from Services.ImageProcessingService import processForEmotions
from Services.ImageProcessingService import modifyPosition

class EmotionService:
	
	def __init__(self):
		# loading models
		path = '../TrainedModels/emotionsClassifier.hdf5'
		self.emotionClassifier = load_model(path, compile=False)
		
		#reshape
		self.sizeTarget = self.emotionClassifier.input_shape[1:3]
	
	def getEmotion(self,coordinates,image):
		x1, x2, y1, y2 = modifyPosition(coordinates)
		face = image[y1:y2, x1:x2]		
		face = cv2.resize(face, (self.sizeTarget))		
		face = processForEmotions(face)				
		emotionLabel = np.argmax(self.emotionClassifier.predict(face))
		return emotionLabel
		
