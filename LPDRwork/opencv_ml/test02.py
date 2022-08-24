import skimage.io
import skimage.color
import skimage.transform
import matplotlib.pyplot as plt

import numpy as np
import os
import random

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

DATASET_DIR = '../images/plate2.jpg'

classes = 1
# classes = os.listdir(DATASET_DIR)
# classes = os.listdir(DATASET_DIR + "/chs_train/")
data = []
for cls in classes:
    # files = os.listdir(DATASET_DIR +os.sep+ cls)
    # files = os.listdir(DATASET_DIR + "/chs_train/" + cls)
    # for f in files:
    #     img = skimage.io.imread(DATASET_DIR + "/chs_train/" + cls + "/" + f)
    img = skimage.io.imread(DATASET_DIR)
    img = skimage.color.rgb2gray(img)
    data.append({
        'x': img,
        'y': cls
    })

random.shuffle(data)
X = [d['x'] for d in data]
y = [d['y'] for d in data]

ys = list(np.unique(y))
y = [ys.index(v) for v in y]

x_train = np.array(X[:int(len(X) * 0.8)])
y_train = np.array(y[:int(len(X) * 0.8)])

x_test = np.array(X[int(len(X) * 0.8):])
y_test = np.array(y[int(len(X) * 0.8):])
batch_size = 128
num_classes = len(classes)
epochs = 10

# input image dimensions
img_rows, img_cols = 20, 20


def extend_channel(data):
    if K.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols, 1)

    return data


x_train = extend_channel(x_train)
x_test = extend_channel(x_test)

input_shape = x_train.shape[1:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train2 = keras.utils.to_categorical(y_train, num_classes)
y_test2 = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train2,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test2))

score = model.evaluate(x_test, y_test2, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save_weights('char_cnn.h5')