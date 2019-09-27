import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#model = load_model('cnn_recaptcha.h5')

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
        plt.show()
        print('plots should have been run')

train_path = '/home/val/google_recaptcha_set/recaptcha_set/3x3/train'
test_path = '/home/val/google_recaptcha_set/recaptcha_set/3x3/test'
valid_path = '/home/val/google_recaptcha_set/recaptcha_set/3x3/valid'

categories = ['bus', 'traffic_lights', 'crosswalks', 'bicycles',
        'fire_hydrant', 'cars', 'chimneys', 'stairs', 'bridges']

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#y_train = np_utils.to_categorical(y_train, num_classes)
#y_test = np_utils.to_categorical(y_test, num_classes)

#X_train, X_test, y_train, y_test = train_test_split(images, labels)

model = Sequential()
model.add(Conv2D(32, input_shape=(130, 130, 3), kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('sigmoid'))


model.add(Dense(10, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(130, 130),
        batch_size=16,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_path,
        target_size=(130, 130),
        batch_size=16,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=20)


pics = []
pic = load_img('./pic_2_23.jpg', target_size=(130, 130, 3), color_mode='rgb')
pic = img_to_array(pic)
pic = pic.reshape((-1, 130, 130, 3))
pics.append(pic)
y_prob = model.predict_classes(pics) 
#y_classes = y_prob.argmax(axis=-1)
print(y_prob)


#model.save_weights('cnn_recaptcha_weights.h5')
#model.save('cnn_recaptcha.h5')
#print(model.summary())




#conf_mat = confusion_matrix(test_labels, predected_labels)

"""
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
   		#time.sleep(15)"""




