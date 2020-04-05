#!/usr/bin/python3
import tensorflow as tf
print("Version: " + tf.__version__)

variable = tf.Variable([[30,20],[10,45]])
constant = tf.constant([[23,40],[32,51]])

session = tf.Session()
session.run(tf.global_variables_initializer())

print(session.run(variable))
print(session.run(constant))
print(variable.eval(session))
print(constant.eval(session))