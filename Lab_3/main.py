import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(233)
np.random.seed(233)

model = Sequential()

model.add(Input(shape=(28,28,1)))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from keras.utils import plot_model

plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=False,rankdir='TB')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000,28,28,1)).astype('float')/255
X_test = X_test.reshape((10000,28,28,1)).astype('float')/255

Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=1, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
history = model.fit(X_train, Y_train, batch_size=128, epochs=5, shuffle=True, verbose=2, validation_split=0.3, callbacks=[reduce_lr])

scores = model.evaluate(X_test,Y_test, batch_size=128, verbose=1)

print('The test loss is %f' % scores[0])
print('The accuracy of the model is %f' % scores[1])

hist = history.history
hist['epoch'] = history.epoch

plt.figure('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(hist['epoch'], hist['loss'], label='train loss')
plt.plot(hist['epoch'], hist['val_loss'], label='val loss')
plt.legend()
plt.figure('model accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.plot(hist['epoch'], hist['accuracy'], label='train accuracy')
plt.plot(hist['epoch'], hist['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
