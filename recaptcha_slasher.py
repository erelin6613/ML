#!/usr/bin/env python3

"""

Developed by Valentyna Fihurska https://github.com/erelin6613

Recaptcha_slasher_v1.1.2

Deployment of CNN trained to classify Google Recaptcha images.
The pipeline set not to just solve Recaptcha but to collect
images further. The accuracy of the model could be
improved (I belive, ~0.7 on test set) nevertheless the specifics
of Recaptcha Service were taken into account even when the 
model predicts insufficiant number of pictures.

During testing and deployment became apparent the necessity
to extend model to classify more categories, while training
images are collected temporary solution should be
random clicking (function random_solver)

Currently building a data set with annotated images
for object detection rather then classification to be
trained afterwars on Google Recaptcha images


Last update: 23-Nov-2019
"""

import requests
from time import gmtime, strftime, sleep, time
import numpy as np
import json
import pyautogui
import autopy
import cv2
from keras.models import load_model
import os
from PIL import Image
import pickle
import random
import matplotlib.pyplot as plt
import os
import skimage
from skimage import data
from skimage.filters import threshold_multiotsu, try_all_threshold
from skimage import io
from imageai.Detection import ObjectDetection

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image='/home/val/google_recaptcha_set/captured_1573138014.png', output_type='array', minimum_percentage_probability=30)
for each in detections[1]:
	print(each['name'], '-', each['percentage_probability'])

"""

(array([[[ 88,  85,  77],
        [ 88,  85,  77],
        [ 88,  85,  77],
        ...,
        [ 88,  85,  77],
        [ 88,  85,  77],
        [ 88,  85,  77]],

       [[ 87,  84,  76],
        [ 87,  84,  76],
        [ 87,  84,  76],
        ...,
        [ 87,  84,  76],
        [ 87,  84,  76],
        [ 87,  84,  76]],

       [[ 86,  83,  75],
        [ 86,  83,  75],
        [ 86,  83,  75],
        ...,
        [ 86,  83,  75],
        [ 86,  83,  75],
        [ 86,  83,  75]],

       ...,

       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]],

       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]],

       [[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8), 
        [{'name': 'refrigerator', 'percentage_probability': 33.30824375152588, 'box_points': [40, 0, 1858, 1080]}, 
        {'name': 'tv', 'percentage_probability': 53.8171112537384, 'box_points': [88, 0, 1852, 1080]}])


"""

#from sets import Set

class_decode = {'traffic light': 9, 'stair': 8, 'palm tree': 7, 'fire hydgrant':6,
				'crosswalk': 5, 'chimney':4, 'cars':3, 'bicycle': 2, 'bus':1, 'bridge': 0}

class_decode_ = ['bridge', 'bus', 'bicycle', 'cars', 'chimney', 'crosswalk', 'fire hydgrant', 'palm tree', 'stair', 'traffic light']

framed_pics = (538, 262, 930, 654)
mini_pic_sizes = [(1, 1, 131, 131), (1, 131, 131, 261), (1, 261, 131, 391), (131, 1, 261, 131), (131, 131, 261, 261), (131, 261, 261, 391), (261, 1, 391, 131), (261, 131, 391, 261), (261, 261, 391, 391)]

path = '/home/val/google_recaptcha_set/recaptcha_set/3x3/pic_1573740277.jpg'

def cut_pictures(path):

	"""
	google recaptcha grid:

		0 | 3 | 6
		1 | 4 | 7
		2 | 5 | 8

	"""

	pics = []
	img = Image.open(path)
	new_image = img.crop(framed_pics)
	for i in range(9):
		fin_img = new_image.crop(mini_pic_sizes[i])
		fin_img.save('/home/val/google_recaptcha_set/pic_{}.jpg'.format(int(time())+i))
		fin_img = np.asarray(fin_img)
		pics.append(np.expand_dims(fin_img, axis=0))
		print(np.expand_dims(fin_img, axis=0).shape)
		i += 1
	os.remove(path)

	return pics




def recognizing_pics(path, category):
	
	pics = cut_pictures(path)
	print(pics)

	model = load_model('cnn_model_v5_no_normalization.h5')
	clicked = set()
	i = 0
	for j in range(3):
		for pic in pics:
			print(model.predict_classes(pic)[0])
			if int(category) == int(model.predict_classes(pic)[0]):

				if mini_pic_sizes[i] not in clicked:
					clicked.add(mini_pic_sizes[i])
					button = pyautogui.position(x=536 + random.randint(mini_pic_sizes[i][0], mini_pic_sizes[i][2]), y=269 + random.randint(mini_pic_sizes[i][1], mini_pic_sizes[i][3]))
					pyautogui.leftClick(button)
					sleep(3)
			if len(clicked) > 5:
				break
			i += 1
		if len(clicked) < 5:
			i = 0
			for pic in pics:
				if mini_pic_sizes[i] not in clicked:
					if model.predict_proba(pic)[0][category] > 0.0001:
						button = pyautogui.position(x=536 + random.randint(mini_pic_sizes[i][0], mini_pic_sizes[i][2]), y=269 + random.randint(mini_pic_sizes[i][1], mini_pic_sizes[i][3]))
						pyautogui.leftClick(button)
						sleep(3)
						clicked.add(mini_pic_sizes[i])
				if len(clicked) > 5:
					break

		verify = pyautogui.position(x=870, y=700)
		pyautogui.leftClick(verify)
		sleep(2)


def recognize_query(path):

	file = open('ocr_api_key.txt', 'r')
	ocr_api_url = 'https://api.ocr.space/parse/image'
	ocr_api_key = file.readlines()[0].strip()

	img = open(path, 'rb')
	r = requests.post(ocr_api_url, files = {'img': img}, data = {'apikey': ocr_api_key, 'language': 'eng'})
	query = r.content.decode()
	for key in class_decode.keys():
		if key in query.strip().lower() or key in query.lower():
			category = key
			print(category)
			requests.post(url=ocr_api_url, data={'apikey': ocr_api_key}, headers={'Connection':'close'})
			break
	try:
		print(class_decode[category])
		recognizing_pics(path, class_decode[category])
	except Exception as e:
		print(e)
		random_solver(path)


def random_solver(path):

	#model = load_model('cnn_model_v5_no_normalization.h5')
	clicked = set()
	#i = 0
	#for pic in pics:
	#	print(model.predict_classes(pic)[0])
	for i in range(5):
		if mini_pic_sizes[i] not in clicked:
			clicked.add(mini_pic_sizes[i])
			button = pyautogui.position(x=536 + random.randint(mini_pic_sizes[i][0], mini_pic_sizes[i][2]), y=270 + random.randint(mini_pic_sizes[i][1], mini_pic_sizes[i][3]))
			pyautogui.leftClick(button)
			sleep(3)
		if len(clicked) > 5:
			break
		i += 1

	if len(clicked) < 5:
		i = 0
		for i in range(5):
			if mini_pic_sizes[i] not in clicked:
				if model.predict_proba(pic)[0][category] > 0.00001:
					button = pyautogui.position(x=536 + random.randint(mini_pic_sizes[i][0], mini_pic_sizes[i][2]), y=270 + random.randint(mini_pic_sizes[i][1], mini_pic_sizes[i][3]))
					pyautogui.leftClick(button)
					sleep(3)
					clicked.add(mini_pic_sizes[i])
			if len(clicked) > 5:
				break
	verify = pyautogui.position(x=870, y=700)
	pyautogui.leftClick(verify)
	sleep(2)

def recaptcha_slasher(i, driver):

	browser = pyautogui.position(x=800, y=320)
	pyautogui.leftClick(browser)
	path = '/home/val/google_recaptcha_set/captured_{}.png'.format(i)
	get_screenshot(path, i)
	submit = pyautogui.position(x=518, y=303)
	query = recognize_query(path)
	pyautogui.leftClick(submit)



def get_screenshot(path, i):

	pyautogui.hotkey('f5')
	sleep(5)
	checkbox = pyautogui.position(x=516, y=212)
	pyautogui.leftClick(checkbox)
	sleep(5)
	image = autopy.bitmap.capture_screen()
	image.save(path)

def object_detector(path):

	image = cv2.imread(path)
	print(image)
	#print(image.mean(image))
	#laplacian = cv2.Laplacian(image, cv2.CV_64F)
	edges = cv2.Canny(image, 75, 75)
	cv2.imshow('original', image)
	cv2.imshow('processed', edges)

	#for i in [11, 21, 31, 51, 71, 91, 111, 131, 151, 171, 191, 211, 231]:


		#image_proc_1 = cv2.adaptiveThreshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, i, 1)
		#image_proc_2 = cv2.adaptiveThreshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, i ,2)
		#image_proc_3 = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),i,255,cv2.THRESH_BINARY)
		#cv2.imshow('original', image)
		#cv2.imshow('processed', image_proc_1)
		#cv2.imshow('processed', image_proc_2)
		#cv2.imshow('processed', image_proc_3[1])
	key = cv2.waitKey(10000)



#if __name__ == '__main__':
	#object_detector(path)

