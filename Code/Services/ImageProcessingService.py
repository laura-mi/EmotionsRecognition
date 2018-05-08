from keras.preprocessing import image
import numpy as np
import cv2

def drawLabels(coordinates, image, text, color, xoffset, yoffset):
	x, y = coordinates[:2]
	[h,w] = image.shape[0:2]
	font_scale = h/1000
	cv2.putText(image, text, (x + xoffset, y + yoffset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

def addBoundingBox(coordinates, image, color):
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

def drawDetails(coordinates,rgb_image,emotion_text,gender_text):
	color = (0, 0, 255)
	addBoundingBox(coordinates, rgb_image, color)
	drawLabels(coordinates, rgb_image, gender_text, color, 0, -20)
	drawLabels(coordinates, rgb_image, emotion_text, color, 0, -50)

def modifyPosition(coordinates, offsets= (0,0)):
    x, y, width, height = coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def processForEmotions(x):
	x = x.astype('float32')
	x = x / 255.0
	x = x - 0.5
	x = x * 2.0		
	x = np.expand_dims(x, 0)
	x = np.expand_dims(x, -1)
	return x
	
def processForGender(x):
	x = x.astype('float32')
	x = x / 255.0
	x = np.expand_dims(x, 0)
	return x

def preprocessForFaces(x):
	x = x.astype('float32')
	x = x / 255.0
	return x