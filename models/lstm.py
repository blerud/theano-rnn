import theano
from theano import tensor as T
import numpy as np

from models import model_util as util

class lstm():
    def __init__(self, in_size, rnn_size, out_size, layers, dropout=0,
                 alpha=0.002, adadelta_params=[0.95, 1e-6], is_train=1):
        self.in_size = in_size
        self.out_size = out_size
        self.layers = layers
        self.alpha = alpha
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.adadelta_params = adadelta_params
        self.is_train = is_train
        self.type = 'lstm'

        num_units = layers * 4
        self.w_i = util.create_weights('w_i', (in_size, rnn_size))
        self.w = util.create_weights('w', (num_units, rnn_size, rnn_size))
        self.r = util.create_weights('r', (num_units, rnn_size, rnn_size))
        self.p = util.create_weights('p', (num_units-layers, rnn_size,))
        self.b = util.create_weights('b', (num_units, rnn_size,))
        self.w_o = util.create_weights('w_o', (rnn_size, out_size))

        self.y_tm1 = util.init_zeros('y_tm1', (rnn_size,), layers)
        self.c_tm1 = util.init_zeros('y_tm1', (rnn_size,), layers)

        self.theano_build()

    def weights(self):
        weights = []
        weights.append(self.w)
        weights.append(self.r)
        weights.append(self.p)
        weights.append(self.b)
        weights.append(self.w_i)
        weights.append(self.w_o)
        return weights

    def load_weights(self, model):
        weights = self.weights()
        for w in weights:
            w.set_value(model[w.name])

    def lstm_layer(self, layer, u, h0, c0):
        def forward(u, y_tm1, c_tm1):
            w,r,p,b = self.w, self.r, self.p, self.b
            ls = 4*layer
            lsp = 3*layer
            z = T.tanh(T.dot(u, w[0+ls]) +
                       T.dot(y_tm1, r[0+ls]) + c_tm1 + b[0+ls])
            f = T.nnet.hard_sigmoid(T.dot(u, w[1+ls]) +
                                    T.dot(y_tm1, r[1+ls]) +
                                    p[0+lsp]*c_tm1 + b[1+ls])
            s = T.nnet.hard_sigmoid(T.dot(u, w[2+ls]) +
                                    T.dot(y_tm1, r[2+ls]) +
                                    p[1+lsp]*c_tm1 + b[2+ls])
            c = s*z + f*c_tm1
            o = T.nnet.hard_sigmoid(T.dot(u, w[3+ls]) +
                                    T.dot(y_tm1, r[3+ls]) +
                                    p[2+lsp]*c + b[3+ls])
            h = T.tanh(c)*o
            return h,c
        [h,c], updates = theano.scan(fn=forward,
                                     sequences=u,
                                     outputs_info=[h0, c0],
                                     allow_gc=False)
        if self.dropout != 0:
            srng = T.shared_randomstreams.RandomStreams()
            drop_mask = srng.binomial(n=1, p=self.dropout, size=h.shape,
                                      dtype=theano.config.floatX)
            hd = T.switch(T.eq(self.is_train, 1), h*drop_mask, h*self.dropout)
            return hd,c
        else:
            return h,c

    def theano_build(self):
        X = T.fmatrix()
        Y = T.fmatrix()

        l1 = T.nnet.relu(T.dot(X, self.w_i))
        layers = [(l1, 0)]
        for l in range(self.layers):
            layers.append(self.lstm_layer(l, layers[-1][0],
                                          self.y_tm1[l],
                                          self.c_tm1[l]))
        hyp = T.nnet.softmax(T.dot(layers[-1][0], self.w_o))

        def sgd(cost, weights, alpha):
            gradient = T.grad(cost=cost, wrt=weights)
            update = []
            for s,g in zip(weights, gradient):
                update.append([s, s - g*alpha])
            return update

        def adadelta(cost, weights, accum_grad, accum_updates):
            gradient = T.grad(cost=cost, wrt=weights)
            update = []
            i = 0
            r = self.adadelta_params[0]
            e = self.adadelta_params[1]
            for s,g in zip(weights, gradient):
                accum_grad[i] = r*accum_grad[i] + (1-r)*T.sqr(g)
                u = T.sqrt(accum_updates[i]+e) / T.sqrt(accum_grad[i]+e) * g
                accum_updates[i] = r*accum_updates[i] + (1-r)*T.sqr(u)
                update.append([s, s - u])
                i += 1
            return update

        y_pred = T.argmax(hyp, axis=1)
        weights = self.weights()
        cost = T.mean(T.nnet.categorical_crossentropy(hyp, Y))
        accum_grad = [0.0]*len(weights)
        accum_updates = [0.0]*len(weights)
        update = adadelta(cost, weights, accum_grad, accum_updates)

        self.train = theano.function(inputs=[X, Y],
                                     outputs=cost,
                                     updates=update,
                                     allow_input_downcast=True)
        self.predict = theano.function(inputs=[X],
                                       outputs=y_pred,
                                       allow_input_downcast=True)

