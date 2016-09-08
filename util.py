import pickle
import sys

import numpy as np

from models import gru
from models import lstm

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

