#!/usr/bin/python3
import tensorflow as tf
import numpy as np
print("Version: " + tf.__version__)

constant = tf.constant([[1,1],[1,1]])
print(constant)
print(constant.shape)
print(constant.numpy())

np_tensor = np.array([[2,2],[2,2]])
constant2 = tf.constant(np_tensor)
print(constant2)
print(constant2.shape)
print(constant2.numpy())

variable = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
print(variable)
print(variable.shape)
print(variable.numpy())

print(variable + 2)
print(variable * 2)
print(np.square(variable))
print(np.sqrt(variable.numpy()))
#print(np.sqrt(variable)) # No sirve
print(np.dot(np.transpose(variable.numpy()), np.square(variable.numpy()))) # WTF!?

variable[0,2].assign(1000.)
print(variable)
print(variable.shape)
print(variable.numpy())

constant1 = tf.constant([[1,2,3],[4,5,6]])
print(np.square(constant1))
print(np.sqrt(constant1)) # No sirve
print(np.dot(np.transpose(constant1), np.square(constant1))) # WTF!?


tensor1 = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
tensor2 = np.square(tensor1)
print(np.dot(tensor1,tensor2))