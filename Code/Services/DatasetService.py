import pandas as pd
import cv2
import numpy as np

def getLabels(name = None): 
	fer = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
	imdb ={0:'woman', 1:'man'}
	if name == 'fer2013':
		return fer
	elif name == 'imdb':
		return imdb
	else:
		return fer,imdb
		
def getFerDataset(image_size):
	ferPath = '../Datasets/fer2013/fer2013.csv'
	data = pd.read_csv(ferPath)
	pixels = data['pixels'].tolist()
	width, height = 48, 48
	faces = []
	for pixel_sequence in pixels:
		face = [int(pixel) for pixel in pixel_sequence.split(' ')]
		face = np.asarray(face).reshape(width, height)
		face = cv2.resize(face.astype('uint8'), image_size)
		faces.append(face.astype('float32'))
	faces = np.asarray(faces)
	faces = np.expand_dims(faces, -1)
	emotions = pd.get_dummies(data['emotion']).as_matrix()
	return faces, emotions
	
#Split the training and the validation set
def splitData(x, y, validation_split=.2):
    total_samples = len(x)
    training_samples = int((1 - validation_split)*total_samples)
    train_x = x[:training_samples]
    train_y = y[:training_samples]
    val_x = x[training_samples:]
    val_y = y[training_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
