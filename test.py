import argparse
import sys
import random

import numpy as np
import theano
from theano import tensor as T

from models import gru
from models import lstm
import util

parser = argparse.ArgumentParser()
parser.add_argument('load', default='')
parser.add_argument('-tl', '--test_length', type=int, default=50)
parser.add_argument('-s', '--seed', default='')
args = parser.parse_args()

rnn_test = util.load_model(args.load)
char_to_ix = rnn_test.char_to_ix
ix_to_char = rnn_test.ix_to_char
chars = char_to_ix.keys()

seq_test = args.seed
if seq_test == '':
    seq_test = random.choice(chars)
for n in xrange(args.test_length):
    seq_test_oh = util.onehot(seq_test, char_to_ix)
    prediction = rnn_test.predict(seq_test_oh)
    seq_test += ix_to_char[prediction[-1]]
print seq_test

