
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
import tensorflow as tf

def uniform_init(dimen_A, dimen_B):
    """uniform initializer"""
    normalized = np.sqrt(6) / (np.sqrt(dimen_A + dimen_B))
    return tf.random_uniform([dimen_A, dimen_B], minval=-normalized, maxval=normalized)

def gaussian_init(dimen_A, dimen_B):
    """ gaussian initializer"""
    return tf.random_normal([dimen_A, dimen_B], 0, 2. / (dimen_A * dimen_B))

def identity_func(x, name):
    """identity function"""
    return x

def get_var_list(scope_name):
    """get the updatable variables in this scope """
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

def sample_prior(dimen=(64, 4)):
    """sample from the normal distribution"""
    return np.random.normal(dimen=dimen)

def layer_output(dimen_A, dimen_B, layer_input, layer_name, activation_function=tf.nn.relu, initialization_function=uniform_init, batch_normed=True, epsilon = 1e-3):
    """implement a batch normalization layer"""
    w = tf.Variable(initialization_function(dimen_A, dimen_B), name="weight")
    if not batch_normed:
        b = tf.Variable(tf.random_normal([dimen_B]), name="bias")
        return activation_function(tf.add(tf.matmul(layer_input, w), b), name=layer_name)        
    pre_output = tf.matmul(layer_input, w)
    batch_mean, batch_var = tf.nn.moments(pre_output,[0])
    scale = tf.Variable(tf.ones([dimen_B]))
    beta = tf.Variable(tf.zeros([dimen_B]))
    layer_output_value = tf.nn.batch_normalization(pre_output, batch_mean, batch_var, beta, scale, epsilon)
    return activation_function(layer_output_value, name=layer_name)

