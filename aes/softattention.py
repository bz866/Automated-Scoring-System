import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
#import tensorflow
from keras import activations, initializers, regularizers
from keras.engine.topology import Layer, InputSpec
import numpy as np
#import theano
# from keras.layers.wrappers import Wrapper, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.recurrent import Recurrent, time_distributed_dense


# Build attention pooling layer
class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
    
    def call(self, x, mask=None):
        '''
        x  (71, ?, 100)
        y  (71, 46, 100)
        att_v, tanh(y) (100,) (71, 46, 100)
        att_v, tanh(y) (1, 100, 1) (71, 46, 100)
        weights (1, 1, 46)
        '''
        #print 'x ',x.shape
        y = K.dot(x, self.att_W)
        #print 'y ',K.int_shape(y)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                self.att_v = K.reshape(self.att_v,(1,K.int_shape(self.att_v)[0],1))
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                #tf_session = K.get_session()
                #print 'att_v, tanh(y)', K.int_shape(self.att_v),K.int_shape(K.tanh(y))
                if K.int_shape(K.tanh(y))[0] == None:
                    self.att_v = K.reshape(self.att_v,(1,K.int_shape(self.att_v)[0],1))
                else:
                    self.att_v = K.reshape(self.att_v,(2272,K.int_shape(self.att_v)[0],1))
                #print 'att_v, tanh(y)', K.int_shape(self.att_v),K.int_shape(K.tanh(y))
                weights = K.batch_dot(self.att_v, K.tanh(y), axes=[0, 2])
                #print 'weights', K.int_shape(weights)
        weights = weights.squeeze()
        weights = K.softmax(weights)
        #x.shape[2] 100
        #weights 1,1,46
        weights = K.reshape(weights,(1,K.int_shape(weights)[2]))
        #print(K.int_shape(weights))
        #print(K.int_shape(K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])))
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = K.sum(out,axis=1)
            print 'out', K.int_shape(out)
        elif self.op == 'attmean':
            out = K.sum(out,axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

