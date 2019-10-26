#!/usr/bin/env python3

"""
Developed by Valentyna Fihurska

The CNN model developed to categorize images collected from
Google Recaptcha service and hence trained with images used
by this service. Currently model`s accuracy is poor. Several
problems could be addressed to improve the model:

Ones I have no or little control over:
1. Google Recaptcha service purposely uses blurry images as
mean of protection from robots
2. Hardware; current hardware does not allow to use GPU and
CPU usage is limited also (16 units per layer overwhealms
PC to use all of the resources available)

Ones I can fix or improve:
1. Use skimage module to make images less blury/add contrast/ect
2. Exclude ambigous images (i.e. containging two categories at once)

Last update: 21-Oct-2019
"""

import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical, normalize
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model
import os
import cv2
import tensorflow as tf

"""config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)"""
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#model = load_model('new_model.h5')

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


def create_data(path):

  data = []
  labels = []

  for folder in os.listdir(path):
    for pic in os.listdir(os.path.join(path,folder)):
      label = str(folder)
      img = os.path.join(os.path.join(path,folder), pic)
      img = cv2.imread(img)
      data.append([np.array(img), label])
      #labels.append(label)

  return data


train_path = './recaptcha_set/3x3/train'
test_path = './recaptcha_set/3x3/test'

#train_path = '/home/val/coding/intel_img/seg_train'
#test_path = '/home/val/coding/intel_img/seg_test'
#valid_path = '/home/val/google_recaptcha_set/recaptcha_set/3x3/valid'

train_data = [row[0] for row in create_data(train_path)]
train_data = np.asarray(train_data)
train_data = normalize(train_data)

train_labels = np.asarray([row[1] for row in create_data(train_path)])
train_labels = np.asarray(train_labels)
#train_labels = to_categorical(train_labels)

test_data = [row[0] for row in create_data(test_path)]
test_data = np.asarray(test_data)
test_data = normalize(test_data)

test_labels = np.asarray([row[1] for row in create_data(test_path)])
test_labels = np.asarray(test_labels)
#test_labels = to_categorical(test_labels)
#test_labels = np.asarray(test_labels)

print(train_labels)
#train_data, train_labels = create_data(train_path)
#test_data, test_labels = create_data(test_path)


label_encoder = LabelEncoder()
int_train_labels = label_encoder.fit_transform(train_labels)
#one_hot_enc = OneHotEncoder()
int_train_labels = to_categorical(int_train_labels)
#one_hot_train_enc = one_hot_train_enc.todense()
int_test_labels = label_encoder.fit_transform(test_labels)
int_test_labels = to_categorical(int_test_labels)
#one_hot_enc = OneHotEncoder()
#one_hot_test_enc = one_hot_enc.fit(int_test_labels.reshape(len(int_test_labels), 1))
#one_hot_test_enc = one_hot_test_enc.todense()

print(int_train_labels)

model = Sequential()
model.add(Conv2D(32, input_shape=(130, 130, 3), kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Activation('sigmoid'))

#model.add(Flatten())
model.add(Dense(10, activation=tf.nn.softmax))
#model.add(Flatten())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, int_train_labels, steps_per_epoch=50, epochs=50, validation_split=0.2, validation_steps=20)

print(model.metrics_names)
print(model.evaluate(test_data, int_test_labels))

model.save('cnn_model_v3.h5')
