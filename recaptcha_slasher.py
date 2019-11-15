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

As the algorythm does not perform good enough another
idea is to overwrite the model not just to recognize but
to detect and recognize objects as big reason of not
all that successful performance is categorical
classification despite pictures having several objects.


Last update: 15-Nov-2019
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
#from sets import Set

class_decode = {'traffic light': 9, 'stair': 8, 'palm tree': 7, 'fire hydgrant':6,
				'crosswalk': 5, 'chimney':4, 'cars':3, 'bicycle': 2, 'bus':1, 'bridge': 0}

class_decode_ = ['bridge', 'bus', 'bicycle', 'cars', 'chimney', 'crosswalk', 'fire hydgrant', 'palm tree', 'stair', 'traffic light']

framed_pics = (538, 262, 930, 654)
mini_pic_sizes = [(1, 1, 131, 131), (1, 131, 131, 261), (1, 261, 131, 391), (131, 1, 261, 131), (131, 131, 261, 261), (131, 261, 261, 391), (261, 1, 391, 131), (261, 131, 391, 261), (261, 261, 391, 391)]

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
		fin_img.save('/home/val/google_recaptcha_set/recaptcha_set/3x3/pic_{}.jpg'.format(int(time())+i))
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
					button = pyautogui.position(x=536 + random.randint(mini_pic_sizes[i][0], mini_pic_sizes[i][2]), y=270 + random.randint(mini_pic_sizes[i][1], mini_pic_sizes[i][3]))
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
						button = pyautogui.position(x=536 + random.randint(mini_pic_sizes[i][0], mini_pic_sizes[i][2]), y=270 + random.randint(mini_pic_sizes[i][1], mini_pic_sizes[i][3]))
						pyautogui.leftClick(button)
						sleep(3)
						clicked.add(mini_pic_sizes[i])
				if len(clicked) > 5:
					break

		verify = pyautogui.position(x=870, y=700)
		pyautogui.leftClick(verify)
		sleep(2)


def recognize_query(path):

	ocr_api_url = 'https://api.ocr.space/parse/image'
	ocr_api_key = '1a1b08f53788957'

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
