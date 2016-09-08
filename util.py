import pickle
import sys

import numpy as np
import theano
from theano import tensor as T

from models import gru
from models import lstm

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

def onehot(sequence, char_to_ix):
    seq_oh = np.zeros((len(sequence), len(char_to_ix)))
    for i,x in enumerate(sequence):
        seq_oh[i][char_to_ix[x]] = 1
    return seq_oh

def save_model(f, model, char_to_ix, ix_to_char):
    ps = {}
    ps['type'] = model.type
    ps['in_size'] = model.in_size
    ps['out_size'] = model.out_size
    ps['layers'] = model.layers
    ps['rnn_size'] = model.rnn_size
    ps['char_to_ix'] = char_to_ix
    ps['ix_to_char'] = ix_to_char
    for w in model.weights():
        ps[w.name] = w.get_value()
    pickle.dump(ps, open(f, 'wb'))

def load_model(f, model='', is_train=1):
    try:
        ps = pickle.load(open(f, 'rb'))
    except:
        print 'file not found'
        sys.exit(1)
    if model == '' and ps['type'] == 'lstm':
        model = lstm.lstm(ps['in_size'], ps['rnn_size'], ps['out_size'],
                          ps['layers'], is_train=is_train)
    elif model == '' and ps['type'] == 'gru':
        model = gru.gru(ps['in_size'], ps['rnn_size'], ps['out_size'],
                        ps['layers'], is_train=is_train)
    model.char_to_ix = ps['char_to_ix']
    model.ix_to_char = ps['ix_to_char']
    model.load_weights(ps)
    return model

