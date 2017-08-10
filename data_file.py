
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
import tensorflow as tf

def gen_different_conc_for_same_fp(unique_fingerprints, num_examples=64, num_difference=1, mu=-5.82, std=1.68):
    """
    Generator of same fingerprints with different concentraition
    """
    
    if num_examples % num_difference: 
        raise ValueError('num_examples {0} must be divisible by num_difference {1}'.format(num_examples, num_difference))
    max_index = unique_fingerprints.shape[0] / num_difference
    targets = np.zeros((num_examples, num_examples))
    block_size = num_examples/num_difference
    for i in range(num_difference):
        """blocks of ones for every block of equal fingerprint"""
        targets[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = 1.
    targets = targets > 0
    
    while 1:
        np.random.shuffle(unique_fingerprints)
        for i in range(max_index):
            batch_concentration = np.random.normal(mu, std, size=(num_examples, 1))
            batch_fp = np.repeat(unique_fingerprints[i*num_difference:(i+1)*num_difference], [block_size]*num_difference, axis=0)
            yield batch_fp, batch_concentration, targets
            
def data_loader(data_file_name, unique_data_file_name):
    """data loader for test, train and fingerprint data"""
    data = np.load(data_file_name)
    # 166 bits fingerprint, 1 concentration float, 1 TGI float
    unique_fp = np.load(unique_data_file_name)
    # there is 6252 unique fingerprints and multiple experiments with each

    np.random.shuffle(data)
    test_data, train_data = np.vsplit(data, [100])

    return test_data, train_data, unique_fp

def batch_gen(data, batch_size=64):
    """A simple generator for batch data"""
    max_index = data.shape[0]/batch_size
    while True:
        np.random.shuffle(data)
        for i in range(max_index):
            yield np.hsplit(data[batch_size*i:batch_size*(i+1)], [-2, -1]) 

