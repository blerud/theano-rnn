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
parser.add_argument('filename')
parser.add_argument('-m', '--model', default='lstm')
parser.add_argument('-l', '--load', default='')
parser.add_argument('-s', '--save', default='model.model')
parser.add_argument('-sl', '--seq_length', type=int, default=50)
parser.add_argument('-rs', '--rnn_size', type=int, default=128)
parser.add_argument('-ls', '--layers', type=int, default=2)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-d', '--dropout', type=float, default=0.0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.002)
parser.add_argument('-add', '--adadelta_decay', type=float, default=0.95)
parser.add_argument('-alr', '--adadelta_lr', type=float, default=1e-6)
parser.add_argument('-tl', '--test_length', type=int, default=50)
args = parser.parse_args()

data = ''
with open(args.filename) as fin:
    data = fin.read()
chars = list(set(data))
data_size, chars_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

in_size = chars_size
out_size = chars_size
rnn_size = args.rnn_size
model_type = args.model
rnn_layers = args.layers
seq_length = args.seq_length
epochs = args.epochs
alpha = args.learning_rate
dropout = args.dropout
adadelta_params = [args.adadelta_decay, args.adadelta_lr]

model_load_file = args.load
model_save_file = args.save

if model_load_file != '':
    rnn = util.load_model(model_load_file)
    rnn.dropout = dropout
    rnn.adadelta_params = adadelta_params
    rnn.alpha = alpha
    char_to_ix = rnn.char_to_ix
    ix_to_char = rnn.ix_to_char
elif model_type == 'lstm':
    rnn = lstm.lstm(in_size, rnn_size, out_size, rnn_layers, dropout=dropout,
                    adadelta_params=adadelta_params, alpha=alpha)
elif model_type == 'gru':
    rnn = gru.gru(in_size, rnn_size, out_size, rnn_layers, dropout=dropout,
                  adadelta_params=adadelta_params, alpha=alpha)

for e in range(epochs):
    p = 0
    costs = []
    while p+seq_length+1 < len(data):
        seq_x = data[p:p+seq_length]
        seq_y = data[p+1:p+seq_length+1]

        seq_x_oh = util.onehot(seq_x, char_to_ix)
        seq_y_oh = util.onehot(seq_y, char_to_ix)

        cost = rnn.train(seq_x_oh, seq_y_oh)
        costs.append(cost)
        p += seq_length
    print('epoch', e, 'cost: ', np.mean(costs))
    util.save_model(model_save_file, rnn, char_to_ix, ix_to_char)
    rnn_test = util.load_model(model_save_file, is_train=0)
    seq_test = random.choice(chars)
    for n in range(args.test_length):
        seq_test_oh = util.onehot(seq_test, char_to_ix)
        prediction = rnn_test.predict(seq_test_oh)
        seq_test += ix_to_char[prediction[-1]]
    print(e, seq_test)

