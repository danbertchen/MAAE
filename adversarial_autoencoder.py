
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
import tensorflow as tf

from .data_file import * 
from .model_utils import *

class AAE(object):
    

    def __init__(self, latent_size=4,input_size=166, encoder_layers=2, decoder_layers=2,discriminative_layers=1,
                 learning_rate=0.01, pretrain_batch_size=512, batch_size=64, initializer=gaussian_init): 
        
        # model
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.discriminative_layers = discriminative_layers 
        self.initializer = initializer
        
        # optimizer
        self.learning_rate = learning_rate  
        self.pretrain_batch_size = pretrain_batch_size
        self.batch_size = batch_size 

        # place holder  
        
        self.fp_tensor = tf.placeholder(tf.float32, [None, input_size])
        self.prior_tensor = tf.placeholder(tf.float32, [None, latent_size])
        self.concentration_tensor = tf.placeholder(tf.float32, [None, 1])
        self.tgi_tensor = tf.placeholder(tf.float32, [None, 1])
        self.targets_tensor = tf.placeholder(tf.bool, [None, None])
        self.visible_tensor = tf.concat(1, [self.fp_tensor, self.concentration_tensor])
        self.hidden_tensor = tf.concat(1, [self.prior_tensor, self.tgi_tensor])

        # setup model, loss and optimizer 
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        
        # load training, testing data and all the unique fingerprint file 
        self.test_data, self.train_data, self.unique_fp = data_loader("mcf7.data.npy", "mcf7.unique.fp.npy")        
        
    def _setup_model(self):
        """define the autoencoder model"""    
        # Encoder: 
        # input:  166+1 
        # layer-1: 128
        # layer-2: 64 
        # latent-layer: 3+1
        
        with tf.name_scope("encoder"):
            
            encoder = [self.visible_tensor]
            sizes = [self.input_size + 1, 128, 64, self.latent_size]

            for layer_number in xrange(encoder_layers):
                with tf.name_scope("encoder-{0}".format(layer_number)):
                    encoder_layer = layer_output(sizes[layer_number], sizes[layer_number + 1], encoder[-1], 'encoder_layer')
                    encoder.append(encoder_layer)

            with tf.name_scope("encoder-fp"):
                self.encoded_fp = layer_output(sizes[-2], sizes[-1],  encoder[-1], 'encoded_fp', batch_normed=False, activation_function=identity_func)

        with tf.name_scope("tgi-encoder"):
            self.encoded_tgi = layer_output(sizes[-2], 1,  encoder[-1], 'encoded_tgi', batch_normed=False, activation_function=identity_func)

        self.encoded = tf.concat(1, [self.encoded_fp, self.encoded_tgi])
        
        # Decoder
        # latent:  3+1
        # layer-1: 64
        # layer-2: 128
        # output:  166+1

        sizes = [self.latent_size + 1, 64, 128, self.input_size]

        with tf.name_scope("decoder"):
            
            decoder = [self.encoded]
            generator = [self.hidden_tensor]

            for layer_number in range(decoder_layers):
                with tf.name_scope("decoder-{0}".format(layer_number)):
                    w = tf.Variable(self.initializer(sizes[layer_number], sizes[layer_number + 1]), name="weight")
                    b = tf.Variable(tf.random_normal([sizes[layer_number + 1]]), name="bias")
                    decoder_layer = tf.nn.tanh(tf.add(tf.matmul(decoder[-1], w), b), name="decoder_layer")
                    gen_layer = tf.nn.tanh(tf.add(tf.matmul(generator[-1], w), b), name="gen_layer")
                    decoder.append(decoder_layer)
                    generator.append(gen_layer)
     
            with tf.name_scope("decoder-fp"):
                w = tf.Variable(self.initializer(sizes[-2], sizes[-1]), name="weight")
                b = tf.Variable(tf.random_normal([sizes[-1]]), name="bias")
                self.decoded_fp = tf.add(tf.matmul(decoder[-1], w), b, name="decoder_fp")
                self.gen_fp = tf.nn.relu(tf.add(tf.matmul(generator[-1], w), b), name="gen_fp")

            with tf.name_scope("decoder_concentration"):
                w = tf.Variable(self.initializer(sizes[-2], 1), name="weight")
                b = tf.Variable(tf.random_normal([1]), name="bias")
                self.decoded_concentration = tf.add(tf.matmul(decoder[-1], w), b)
                self.gen_concentration = tf.add(tf.matmul(generator[-1], w), b)

        # Discriminative net: 
        # latent: 3
        # layer-1: 64
        # layer-2: 3
        # output: 1 
        
        with tf.name_scope("discriminator"):
            
            discriminator_enc = [self.encoded_fp]
            discriminator_prior = [self.prior_tensor]
            sizes = [self.latent_size, 2 * self.latent_size - 2, 1]

            for layer_number in xrange(discriminative_layers):
                with tf.name_scope("discriminator-{0}".format(layer_number)):
                    w = tf.Variable(self.initializer(sizes[layer_number], sizes[layer_number + 1]), name="weight")
                    b = tf.Variable(tf.random_normal([sizes[layer_number + 1]]), name="bias")
                    discriminative_enc = tf.nn.relu(tf.add(tf.matmul(discriminator_enc[-1], w), b), name="discriminative_enc")
                    discriminative_prior = tf.nn.relu(tf.add(tf.matmul(discriminator_prior[-1], w), b), name="discriminative_prior")      
                    discriminator_enc.append(discriminative_enc)
                    discriminator_prior.append(discriminative_prior)

            with tf.name_scope("discriminator-final"):
                w = tf.Variable(self.initializer(sizes[-2], sizes[-1]), name="weight")
                b = tf.Variable(tf.random_normal([sizes[-1]]), name="bias")
                self.discriminative_enc = tf.add(tf.matmul(discriminator_enc[-1], w), b, name="discriminative_enc")
                self.discriminative_prior = tf.add(tf.matmul(discriminator_prior[-1], w), b, name="discriminative_prior")
            
            
    def _setup_loss(self):
        """define various loss function"""
        # discriminative loss 
        self.discriminative_loss = tf.reduce_mean(tf.nn.relu(self.discriminative_prior) - self.discriminative_prior + tf.log(1.0 + tf.exp(-tf.abs(self.discriminative_prior)))) +                                    tf.reduce_mean(tf.nn.relu(self.discriminative_enc) + tf.log(1.0 + tf.exp(-tf.abs(self.discriminative_enc))))
            
        fp_norms = tf.sqrt(tf.reduce_sum(tf.square(self.encoded_fp), keep_dims=True, reduction_indices=[1]))
        normalized_fp = tf.div(self.encoded_fp, fp_norms)
        cosines_fp = tf.matmul(normalized_fp, tf.transpose(normalized_fp))
        self.manifold_cost = tf.reduce_mean(1 - tf.boolean_mask(cosines_fp, self.targets_tensor))
       
        # encoder loss 
        self.encoder_fp_loss = tf.reduce_mean(tf.nn.relu(self.discriminative_enc) - self.discriminative_enc + tf.log(1.0 + tf.exp(-tf.abs(self.discriminative_enc))))
        self.encoder_tgi_loss = tf.reduce_mean(tf.square(tf.sub(self.tgi_tensor, self.encoded_tgi)))
        self.encoder_loss = self.encoder_fp_loss

        # cross entropy for fingerprint reconstruction error; mean square error for concentration part 
        self.decoder_fp_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.decoded_fp, self.fp_tensor))
        self.decoder_concentration_loss = tf.reduce_mean(tf.square(tf.sub(self.concentration_tensor, self.decoded_concentration)))
        self.decoder_loss = self.decoder_fp_loss + self.decoder_concentration_loss
        
    def _setup_optimizer(self):　
        """define various training optimizer"""
        self.train_discriminator = tf.train.AdamOptimizer(self.learning_rate).minimize(self.discriminative_loss, var_list=get_var_list('discriminator'))
        self.train_encoder = tf.train.AdamOptimizer(self.learning_rate).minimize(self.encoder_loss, var_list=get_var_list('encoder'))
        self.train_manifold = tf.train.AdamOptimizer(self.learning_rate).minimize(self.manifold_cost, var_list=get_var_list('encoder'))
        self.train_reg = tf.train.AdamOptimizer(self.learning_rate).minimize(self.encoder_tgi_loss, var_list=get_var_list('encoder') + get_var_list('tgi-encoder'))
        self.train_autoencoder = tf.train.AdamOptimizer(self.learning_rate).minimize(self.decoder_loss, var_list=get_var_list('encoder') + get_var_list('tgi-encoder') + get_var_list('decoder'))
            
        
    def train(self):
        """training the model"""
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        saver = tf.train.Saver()
        same_fp_different_conc = gen_different_conc_for_same_fp(self.unique_fp, num_difference=32)
        self.pretrain(self.train_data, sess, init, same_fp_different_conc)
        
        batches = batch_gen(self.train_data, self.batch_size)
        same_fp_different_conc = gen_different_conc_for_same_fp(self.unique_fp, num_difference=32)
        
        for i_epoch in range(50):
            if i_epoch > 0 and i_epoch % 10 == 0:
                saver.save(sess, "./adam.adversial_auto_encoder.manifold.{0:d}e.model.ckpt".format(i_epoch))
            
            print("epoch # {0:d}".format(i_epoch))
            
            for u in range(10000):
                batch_fp, batch_concentration, batch_tgi = batches.next()
                batch_prior = sample_prior()
                sess.run(self.train_discriminator, feed_dict={self.fp_tensor: batch_fp,
                                                         self.concentration_tensor: batch_concentration,
                                                         self.tgi_tensor: batch_tgi,
                                                         self.prior_tensor: batch_prior})

                batch_fp, batch_concentration, batch_tgi = batches.next()

                sess.run(self.train_encoder, feed_dict={self.fp_tensor: batch_fp,
                                                   self.concentration_tensor: batch_concentration})

            
                batch_fp, batch_concentration, batch_tgi = batches.next()
                sess.run(self.train_reg, feed_dict={self.fp_tensor: batch_fp,
                                               self.concentration_tensor: batch_concentration,
                                               self.tgi_tensor: batch_tgi})

                batch_fp, batch_concentration, batch_tgi = batches.next()
                sess.run(self.train_autoencoder, feed_dict={self.fp_tensor: batch_fp,
                                                       self.concentration_tensor: batch_concentration,
                                                       self.tgi_tensor: batch_tgi})
                
            else:
                batch_prior = sample_prior((100, self.latent_size))
                losses = sess.run([self.discriminative_loss, self.encoder_fp_loss, self.encoder_tgi_loss, self.decoder_fp_loss, self.decoder_concentration_loss],
                                  feed_dict={self.fp_tensor: self.train_data[:, :-2],
                                             self.concentration_tensor: self.train_data[:, -2:-1],
                                             self.tgi_tensor: self.train_data[:, -1:],
                                             self.prior_tensor: batch_prior
                                            })
                
                same_fp, same_conc, targets = same_fp_different_conc.next()
                m_loss = sess.run(self.manifold_cost, feed_dict={self.fp_tensor: batch_fp,
                                                            self.concentration_tensor: batch_concentration,
                                                            self.targets_tensor: targets})
                
                discriminator_loss, encoder_fp_loss, encoder_tgi_loss, autoencoder_fp_loss, autoencoder_concentration_loss = losses
                print("disc: {0:f}, encoder_fp : {1:f}, mani_fp: {2:f}, encoder_tgi: {3:f}, decoder_fp : {4:f}, decoder_concentration : {5:f}".format(discriminator_loss/2.,                      encoder_fp_loss,m_loss,encoder_tgi_loss,autoencoder_fp_loss,autoencoder_concentration_loss))

    def pretrain(self, train_data, sess, init, same_fp_different_conc):
        """pretrain generator w/o regressions and decoding"""

        batches = batch_gen(train_data, self.pretrain_batch_size)
        
        flag = True
        while flag:
            
            # try a number of different initializations. 
            # gan is not so stable.  
            # generator doesn't converge. 
            
            sess.run(init)
            for i_epoch in range(15):
                print("epoch #{0:d}".format(e))
                discriminator_loss = 0.0
                encoder_fp_loss = 0.0
                mani_loss = 0.0
                for u in range(1000):
                    batch_fp, batch_concentration, _ = batches.next()
                    batch_prior = sample_prior()
                    _, loss = sess.run([self.train_discriminator, self.discriminative_loss], 
                        feed_dict={self.fp_tensor: batch_fp,
                            self.concentration_tensor: batch_concentration,
                            self.prior_tensor: batch_prior})
                    discriminator_loss += loss

                    fp_loss = 2.
                    count = 0
                    while fp_loss > 1. and count < 20:
                        batch_fp, batch_concentration, _ = batches.next()
                        _, fp_loss = sess.run([self.train_encoder, self.encoder_fp_loss], 
                                            feed_dict={self.fp_tensor: batch_fp,
                                            self.concentration_tensor: batch_concentration,})
                        count += 1
                    else:
                        encoder_fp_loss += fp_loss

                    same_fp, same_conc, targets = same_fp_different_conc.next()
                    _, m_loss = sess.run([self.train_manifold, self.manifold_cost], 
                        feed_dict={self.fp_tensor: batch_fp,
                        self.concentration_tensor: batch_concentration,
                        self.targets_tensor: targets})
                    
                    mani_loss += m_loss

                discriminator_loss /= 1000. * 2.
                encoder_fp_loss /= 1000.
                mani_loss /= 1000.

                print("disc: {0:f}, encoder_p: {1:f}, manifold: {2:f}".format(discriminator_loss, encoder_fp_loss, mani_loss))
                if (i_epoch >= 5) and (encoder_fp_loss < 0.7):
                    flag = False
                    break


