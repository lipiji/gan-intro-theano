#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from updates import *
import matplotlib.pyplot as plt
from matplotlib import animation

seed = 1234
np.random.seed(seed)

class DataDistribution():
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        samples = np.asmatrix(samples)
        samples = samples.reshape((N, 1))
        return floatX(samples)


class GeneratorDistribution():
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        samples = np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
        samples = np.asmatrix(samples)
        samples = samples.reshape((N, 1))
        return floatX(samples)

class GAN():
    def __init__(self, optimizer, in_size, out_size, hidden_size):
        self.X = T.matrix("X")
        self.Z = T.matrix("Z")
        self.optimizer = optimizer
        #self.batch_size = T.iscalar('batch_size')
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        
        self.define_train_test_funcs()

    class Generator():
        def __init__(self, in_size, out_size, hidden_size):
            prefix = "gen_"
            self.in_size = in_size
            self.out_size = out_size
            self.hidden_size = hidden_size
            self.Wg_zh = init_weights((self.in_size, self.hidden_size), prefix + "Wg_zh")
            self.bg_zh = init_bias(self.hidden_size, prefix + "bg_zh")
            self.Wg_hy = init_weights((self.hidden_size, self.in_size), prefix + "Wg_hy")
            self.bg_hy = init_bias(self.in_size, prefix + "bg_hy")
            self.params = [self.Wg_zh, self.bg_zh, self.Wg_hy, self.bg_hy]
        
        def generate(self, z):
            h = T.tanh(T.dot(z, self.Wg_zh) + self.bg_zh)
            y = T.dot(h, self.Wg_hy) + self.bg_hy
            return y

    class Discriminator():
        def __init__(self, in_size, out_size, hidden_size):
            prefix = "dis_"
            self.in_size = in_size
            self.out_size = out_size
            self.hidden_size = hidden_size
            self.Wd_xh = init_weights((self.in_size, self.hidden_size), prefix + "Wd_xh")
            self.bd_xh = init_bias(self.hidden_size, prefix + "bd_xh")
            self.Wd_hh1 = init_weights((self.hidden_size, self.hidden_size), prefix + "Wd_hh1")
            self.bd_hh1 = init_bias(self.hidden_size, prefix + "bd_hh1")
            self.Wd_hh2 = init_weights((self.hidden_size, self.hidden_size), prefix + "Wd_hh12")
            self.bd_hh2 = init_bias(self.hidden_size, prefix + "bd_hh2")
            self.Wd_hy = init_weights((self.hidden_size, self.out_size), prefix + "Wd_hy")
            self.bd_hy = init_bias(self.out_size, prefix + "bd_hy")
            self.params = [self.Wd_xh, self.bd_xh, self.Wd_hh1, self.bd_hh1,\
                                self.Wd_hh2, self.bd_hh2, self.Wd_hy, self.bd_hy]
        def discriminate(self, x):
            h0 = T.tanh(T.dot(x, self.Wd_xh) + self.bd_xh)
            h1 = T.tanh(T.dot(h0, self.Wd_hh1) + self.bd_hh1)
            h2 = T.tanh(T.dot(h1, self.Wd_hh2) + self.bd_hh2)
            y = T.nnet.sigmoid(T.dot(h2, self.Wd_hy) + self.bd_hy)
            return y

    def define_train_test_funcs(self):
        G = self.Generator(self.in_size, self.out_size, self.hidden_size)
        D = self.Discriminator(self.in_size, self.out_size, self.hidden_size)
        self.params_dis = D.params
        self.params_gen = G.params

        g = G.generate(self.Z)
        d1 = D.discriminate(self.X)
        d2 = D.discriminate(g)

        loss_d = T.mean(-T.log(d1) - T.log(1 - d2))
        gparams_d = []
        for param in self.params_dis:
            gparam = T.grad(loss_d, param)
            #gparam = T.clip(T.grad(loss_d, param), -10, 10)
            gparams_d.append(gparam)

        loss_g = T.mean(-T.log(d2))
        gparams_g = []
        for param in self.params_gen:
            gparam = T.grad(loss_g, param)
            #gparam = T.clip(T.grad(loss_g, param), -10, 10)
            gparams_g.append(gparam)

        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params_dis + self.params_gen, gparams_d + gparams_g, lr)
 
        self.train = theano.function(inputs = [self.X, self.Z, lr],
                outputs = [loss_d, loss_g], updates = updates)
           

def main():
    use_gpu(0)
    lr = 1e-4
    batch_size = 1
    hidden_size = 10
    # try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
    optimizer = "adam" 

    datar = DataDistribution()
    gener = GeneratorDistribution(8)

    in_size = 1
    out_size = 1

    print "compiling..."
    model = GAN(optimizer, in_size, out_size, hidden_size)

    print "training..."
    for epoch in xrange(1000):
        x = datar.sample(batch_size) 
        z = gener.sample(batch_size)

        loss_d, loss_g = model.train(x, z, lr)
        print "loss_d = ", loss_d, ", ",  "loss_g = ", loss_g

if __name__ == "__main__":
    main()
