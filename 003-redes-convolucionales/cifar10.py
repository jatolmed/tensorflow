#!/usr/bin/python3
# https://colab.research.google.com/drive/1ltyNUt415iUOsLN-vFFidb9ZeundIOC3#scrollTo=D3MHvRYKe9fN
import tensorflow as tf
#import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.datasets import cifar10

# Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
# https://github.com/tensorflow/tensorflow/issues/24496
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#%matplotlib inline
tf.__version__

# Configurar el nombre de las clases del dataset
class_names = ['avión', 'coche', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Cargar el dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

#plt.imshow(X_test[10])

# Definición del modelo
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Dropout(1./4.))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Dropout(1./4.))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Dropout(1./4.))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dropout(1./4.))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.summary()
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999), metrics=["sparse_categorical_accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=20)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy: {}".format(test_accuracy))