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
X_train = X_train / 255.0
X_test = X_test / 255.0

# Se redimensiona el dataset
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Se crea el modelo
model = tf.keras.models.Sequential()

# Capas ocultas y de dropout
model.add(tf.keras.layers.Dense(units=1176, activation='tanh', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(0.225))
model.add(tf.keras.layers.Dense(units=392, activation='relu', input_shape=(1176,)))
model.add(tf.keras.layers.Dropout(0.275))
model.add(tf.keras.layers.Dense(units=588, activation='tanh', input_shape=(392,)))
model.add(tf.keras.layers.Dropout(0.225))
model.add(tf.keras.layers.Dense(units=196, activation='relu', input_shape=(588,)))
model.add(tf.keras.layers.Dropout(0.275))
model.add(tf.keras.layers.Dense(units=294, activation='tanh', input_shape=(196,)))
model.add(tf.keras.layers.Dropout(0.225))
model.add(tf.keras.layers.Dense(units=98, activation='relu', input_shape=(294,)))
model.add(tf.keras.layers.Dropout(0.275))
model.add(tf.keras.layers.Dense(units=147, activation='tanh', input_shape=(98,)))
model.add(tf.keras.layers.Dropout(0.225))
model.add(tf.keras.layers.Dense(units=49, activation='relu', input_shape=(147,)))
model.add(tf.keras.layers.Dropout(0.275))

# Capa de salida
model.add(tf.keras.layers.Dense(units=10, activation='softmax', input_shape=(49,)))

# Compilación
model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=0.75), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# Entrenamiento
history = model.fit(X_train, y_train, epochs=25)

# Evaluación
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy: {}".format(test_accuracy))

# Se guarda el modelo (topología)
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)

# Se guardan los pesos
model.save_weights("fashion_model.h5")