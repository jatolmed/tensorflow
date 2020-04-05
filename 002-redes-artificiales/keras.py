#!/usr/bin/python3
# https://colab.research.google.com/drive/1O8zlf4KkiExWw5qVS7FhfTCBurbsy6mP#scrollTo=8dQOL_EtChrN
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
print(tf.__version__)

# Se cargan las imágenes
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Se normalizan las imágenes
X_train = X_train / 255. - .5
X_test = X_test / 255. - .5

# Se redimensiona el dataset
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Se crea el modelo
model = tf.keras.models.Sequential()

# Capas ocultas y de dropout
model.add(tf.keras.layers.GaussianNoise(1./256., input_shape=(784,)))
#model.add(tf.keras.layers.Dense(units=112, activation='tanh', input_shape=(784,)))
#model.add(tf.keras.layers.Dropout(1./7.))
#model.add(tf.keras.layers.Dense(units=32, activation='relu', input_shape=(112,)))
#model.add(tf.keras.layers.Dropout(1./16.))
#model.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(32,)))
#model.add(tf.keras.layers.Dropout(1./32.))
#model.add(tf.keras.layers.Dense(units=1024, activation='sigmoid', input_shape=(256,)))
#model.add(tf.keras.layers.Dropout(1./8.))
model.add(tf.keras.layers.Dense(units=1024, activation='tanh', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(1./8.))
model.add(tf.keras.layers.Dense(units=256, activation='tanh', input_shape=(1024,)))
model.add(tf.keras.layers.Dropout(1./32.))
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid', input_shape=(256,)))
model.add(tf.keras.layers.Dropout(1./16.))
model.add(tf.keras.layers.Dense(units=8, activation='relu', input_shape=(32,)))

# Capa de salida
model.add(tf.keras.layers.Dense(units=10, activation='softmax', input_shape=(8,)))

# Compilación
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# Entrenamiento y evaluación
history = model.fit(X_train, y_train, epochs=20)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("#################################################################")
print("##                        20 ITERATIONS                        ##")
print("#################################################################")
print("Test accuracy: {}".format(test_accuracy))

history = model.fit(X_train, y_train, epochs=20)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("#################################################################")
print("##                        40 ITERATIONS                        ##")
print("#################################################################")
print("Test accuracy: {}".format(test_accuracy))

history = model.fit(X_train, y_train, epochs=20)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("#################################################################")
print("##                        60 ITERATIONS                        ##")
print("#################################################################")
print("Test accuracy: {}".format(test_accuracy))

history = model.fit(X_train, y_train, epochs=20)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("#################################################################")
print("##                        80 ITERATIONS                        ##")
print("#################################################################")
print("Test accuracy: {}".format(test_accuracy))

history = model.fit(X_train, y_train, epochs=20)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("#################################################################")
print("##                       100 ITERATIONS                        ##")
print("#################################################################")
print("Test accuracy: {}".format(test_accuracy))

# Se guarda el modelo (topología)
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)

# Se guardan los pesos
model.save_weights("fashion_model.h5")