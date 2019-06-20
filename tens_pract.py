from __future__ import absolute_import, division, print_function
from time import sleep
import cv2
import numpy as np
import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow
import os
import pathlib
from tqdm import tqdm
import random




cell_images = []
img_size = 100

datadir = '/home/val/Downloads/cell_images/'
categories = ['Parasitized', 'Uninfected']

train = []
#test = []

for category in categories:
	path = os.path.join(datadir, category)
	cat = categories.index(category)
	for image in tqdm(os.listdir(path)):
		try:
			img_array = cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)
			new_array = cv2.resize(img_array, (img_size, img_size))
			train.append([new_array, cat])
		#print(img_array.shape)
		#plt.imshow(img_array, cmap='gray')
		#plt.show()
		except Exception as e:
			print(e)
			pass


print(train)
#print(train.shape)
#sleep(60)

random.shuffle(train)

X = []
y = []

for features,label in train:
    X.append(features)
    y.append(label)

print(X)
print(y)
#sleep(30)
X = np.array(X).reshape(-1, img_size, img_size, 1)

X = X/255.0



#dir1 = '/home/val/Downloads/cell_images/Parasitized/'
#dir2 = '/home/val/Downloads/cell_images/Uninfected/'

#for name in os.listdir(dir1):
#	print(name)
#	image = get_an_image(fname = (dir1 + name))
#	plt.imshow()
#	sleep(15)

#X = train[0]
#y = train[1]

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Flatten(),
    #tensorflow.keras.layers.Conv2D(64, (3, 3), activation = tensorflow.nn.relu),
    tensorflow.keras.layers.Dense(128, activation = tensorflow.nn.relu),
    tensorflow.keras.layers.Dense(10, activation = tensorflow.nn.sigmoid)
    ])


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X, y, epochs=3, validation_split=0.3)

#model.predict('')