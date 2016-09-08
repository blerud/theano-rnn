import theano
from theano import tensor as T
import numpy as np

import model_util as util

class gru():
    def __init__(self, in_size, rnn_size, out_size, layers,
                 dropout=0, alpha=0.002, adadelta_params=[0.95, 1e-6], is_train=1):
        self.in_size = in_size
        self.out_size = out_size
        self.layers = layers
        self.alpha = alpha
        self.rnn_size = rnn_size
        self.dropout = dropout
        self.adadelta_params = adadelta_params
        self.is_train = is_train
        self.type = 'gru'

        num_units = layers * 3
        self.w_i = util.create_weights('w_i', (in_size, rnn_size))
        self.e = util.init_weights('r', (rnn_size, rnn_size), num_units)
        self.w = util.init_weights('w', (rnn_size, rnn_size), num_units)
        self.b = util.init_weights('b', (rnn_size,), num_units)
        self.w_o = util.create_weights('w_o', (rnn_size, out_size))

        self.s0 = util.init_zeros('s0', (rnn_size,), layers)

        self.theano_build()

    def weights(self):
        weights = []
        weights.extend(self.e)
        weights.extend(self.w)
        weights.extend(self.b)
        weights.append(self.w_i)
        weights.append(self.w_o)
        return weights

    def load_weights(self, model):
        weights = self.weights()
        for w in weights:
            w.set_value(model[w.name])

    def gru_layer(self, layer, u, s0):
        def forward(u, s_tm1):
            e,w,b = self.e, self.w, self.b
            ls = 3*layer
            z = T.nnet.hard_sigmoid(T.dot(u, e[0+ls]) +
                                    T.dot(s_tm1, w[0+ls]) +
                                    b[0+ls])
            r = T.nnet.hard_sigmoid(T.dot(u, e[1+ls]) +
                                    T.dot(s_tm1, w[1+ls]) +
                                    b[1+ls])
            h = T.tanh(T.dot(u, e[2+ls]) +
                       T.dot(s_tm1*r, w[2+ls]) +
                       b[2+ls])
            s = (T.ones_like(z) - z)*h + z*s_tm1
            return h,s
        [h,s], updates = theano.scan(fn=forward,
                                     sequences=u,
                                     outputs_info=[None, s0],
                                     allow_gc=False)
        if self.dropout != 0:
            srng = T.shared_randomstreams.RandomStreams()
            drop_mask = srng.binomial(n=1, p=self.dropout, size=h.shape,
                                      dtype=theano.config.floatX)
            hd = T.switch(T.eq(self.is_train, 1), h*drop_mask, h*self.dropout)
            return hd,s
        else:
            return h,s

    def theano_build(self):
        X = T.fmatrix()
        Y = T.fmatrix()

        l1 = T.nnet.relu(T.dot(X, self.w_i))
        layers = [(l1, 0)]
        for l in xrange(self.layers):
            layers.append(self.gru_layer(l, layers[-1][0],
                                         self.s0[l]))
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

