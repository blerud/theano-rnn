import numpy as np
import theano
from theano import tensor as T

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def create_weights(name, shape):
    return theano.shared(name=name, value=floatX(np.random.randn(*shape) * 0.001))

def init_weights(name, shape, num):
    return [create_weights(name, shape) for i in xrange(num)]

def create_zeros(name, shape):
    return theano.shared(name=name, value=np.zeros(shape, dtype=theano.config.floatX))

def init_zeros(name, shape, num):
    return [create_zeros(name, shape) for i in xrange(num)]
