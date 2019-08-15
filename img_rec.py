import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

path = '/home/val/google_recaptcha_set/recaptcha_set/3x3'


categories = ['bus', 'traffic lights', 'crosswalks', 'bicycles',
				'a fire hydrant', 'cars', 'chimneys',
				'parking meters', 'palm trees', 'stairs', 'bridges']


model = Sequential()
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(len(categories)))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 16
train_datagen = ImageDataGenerator(rotation_range=0, width_shift_range=0.25,
									height_shift_range=0.25, rescale=1./255, 
									shear_range=0.2, zoom_range=0.15, 
									horizontal_flip=True, fill_mode='nearest')
#for root, dirs, files in os.walk(path):
#   for name in files:
#   	image = load_img(os.path.join(root, name))
#   	x = img_to_array(image)
#   	x = x.reshape((1,)+x.shape)
#   	model.predict(x)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(path+'/train', target_size=(131, 131), 
														batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(path+'/validation', target_size=(131, 131),
															batch_size=batch_size, class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=2000//batch_size, epochs=50,
					validation_data=validation_generator, validation_steps=800//batch_size)

model.save_weights('model_recaptcha.h5')

j=0

#mini_pic_sizes = [(0, 0, 131, 131), (0, 131, 131, 262), (0, 262, 131, 392),
#					(131, 0, 262, 131), (131, 131, 262, 262), (131, 262, 262, 392),
#					(262, 0, 392, 131), (262, 131, 392, 262), (262, 262, 392, 392)]

categories = ['bus', 'traffic lights', 'crosswalks', 'bicycles',
				'a fire hydrant', 'cars', 'chimneys',
				'parking meters', 'palm trees', 'stairs', 'bridges']

#for root, dirs, files in os.walk(path):
#   for name in files:
#   	image = load_img(os.path.join(root, name))
#   	x = img_to_array(image)
#   	x = x.reshape((1,)+x.shape)
#   	model.predict(x)
   	#print(x.shape)
   	#i=0
   	#for batch in datagen.flow(x, batch_size=1, save_to_dir=path, save_prefix='pic', save_format='jpg'):
   	#	i += 1
   	#	if i > 5:
   	#		break
   	

   	#img = Image.open(os.path.join(root, name))
   	#new_image = img.crop((538, 235, 930, 627))
   	#for i in range(9):
   	#	fin_img = new_image.crop(mini_pic_sizes[i])
   	#	fin_img.save('/home/val/google_recaptcha_set/recaptcha_set/3x3/pic_{}.jpg'.format(j))
   		#fin_img.show()
   	#	j += 1
   		#time.sleep(15)




