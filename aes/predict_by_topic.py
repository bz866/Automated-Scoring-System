from __future__ import print_function
import twitter
from collections import OrderedDict
import numpy
import theano
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import six.moves.cPickle as pickle
import sys
import theano.tensor as tensor
import os
import pickle
import twitter
import predict

def prepare_dat(seqs, length, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same length: the length of the
    longest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = length
#     print(lengths)
#     print(seqs)

    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
        lengths = new_lengths
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    
    for idx, s in enumerate(seqs):
        #print('idx is %s' %idx)
        #print('seq is %s' % s)
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask

def prepare_single_seq(seq, length):
    """Create the matrices from single sentence.
    """

    if length < 1:
        return None, None, None

    n_samples = 1
    maxlen = len(seq)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    x[:length, 0] = seq
    x_mask[:length, 0] = 1.
    return x, x_mask

def pred_single_seq(f_pred_prob, prepare_data, data, iterator,verbose=False):
    
    n_samples = 1
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0
    lengths = len(data)

    x, mask = prepare_single_seq(data, lengths)
    #x, mask= prepare_dat([data[t] for t in range(len(data))],lengths,maxlen=None)
    pred_probs = f_pred_prob(x, mask)

    return pred_probs

def pred_p(f_pred_prob, prepare_data, data, iterator,verbose=False):
    """
    If you want to use a trained model, this is useful to compute
    the probabilities of new examples.

    :param f_pred_prob: Theano fct computing the prediction probabilities
    :param prepare_data: usual prepare_data for that dataset.
    :param data: train, or valid tuple (x, y)
    :param iterator: batch of indices
    :param verbose: print status update

    """
    n_samples = len(data)
    #print(n_samples)
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0
    lengths = predict.leng(data)

   
    x, mask= prepare_dat([data[t] for t in range(len(data))],lengths,maxlen=None)
    pred_probs = f_pred_prob(x, mask)
    if verbose:
        print('%d/%d samples classified' % (n_done, n_samples))
        print(probs)

    return pred_probs

def generate_list(model_options, data, single=False):
    model_options = model_options
    ydim = 2
    model_options['ydim'] = ydim
    valid_batch_size = 64
    batch_size = 16

    # init weight vectors
    params = predict.init_params(model_options)

    #load data
    smp = data
    print('loading model')
    params = predict.load_params('lstm_model.npz', params)
    print('finished!')
    tparams = predict.init_tparams(params)
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = predict.build_model(tparams, model_options)

 
    #The first one is the probability that the input sentence is negative, the second is the probability the input is positive
    if single:
        kf_test = predict.get_minibatches_idx(len(smp), valid_batch_size)
        smp_prob = pred_single_seq(f_pred_prob, prepare_dat, smp, kf_test)
    else:
        kf_test = predict.get_minibatches_idx(len(smp[0]), valid_batch_size)
        smp_prob = pred_p(f_pred_prob, prepare_dat, smp, kf_test)

    return smp_prob
