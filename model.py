""" Users write their original chain on to this file.

Chain is a "chain" composed of Function/Link object and Variables.

Fucntion is a callable object not to be trained.
Link is a callable object to be trained.
(note: "callable object" is defined as an object with :meth:__call__)

"""
import chainer
from chainer import initializers

# core unit of Chain
import chainer.functions as F
import chainer.links as L

import numpy

class UNET128(chainer.Chain):
    """

    :cvar insize int: size of first layer
    :cvar initializer chainer.initializer: initializer instance

    """

    insize = 128
    initializer = initializers.HeNormal(scale=1 / numpy.sqrt(2))

    def __init__(self):
        """instance initialization

        note: self.initializer is not specified here because HeNormal
        initializer is the default initializer

        """

        super().__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(4, 32, 3, 1, 1)
            self.c1 = L.Convolution2D(32, 64, 4, 2, 1)
            self.c2 = L.Convolution2D(64, 64, 3, 1, 1)
            self.c3 = L.Convolution2D(64, 128, 4, 2, 1)
            self.c4 = L.Convolution2D(128, 128, 3, 1, 1)
            self.c5 = L.Convolution2D(128, 256, 4, 2, 1)
            self.c6 = L.Convolution2D(256, 256, 3, 1, 1)
            self.c7 = L.Convolution2D(256, 512, 4, 2, 1)
            self.c8 = L.Convolution2D(512, 512, 3, 1, 1)

            self.dc8 = L.Deconvolution2D(1024, 512, 4, 2, 1)
            self.dc7 = L.Convolution2D(512, 256, 3, 1, 1)
            self.dc6 = L.Deconvolution2D(512, 256, 4, 2, 1)
            self.dc5 = L.Convolution2D(256, 128, 3, 1, 1)
            self.dc4 = L.Deconvolution2D(256, 128, 4, 2, 1)
            self.dc3 = L.Convolution2D(128, 64, 3, 1, 1)
            self.dc2 = L.Deconvolution2D(128, 64, 4, 2, 1)
            self.dc1 = L.Convolution2D(64, 32, 3, 1, 1)
            self.dc0 = L.Convolution2D(64, 3, 3, 1, 1)

            self.bnc0 = L.BatchNormalization(32)
            self.bnc1 = L.BatchNormalization(64)
            self.bnc2 = L.BatchNormalization(64)
            self.bnc3 = L.BatchNormalization(128)
            self.bnc4 = L.BatchNormalization(128)
            self.bnc5 = L.BatchNormalization(256)
            self.bnc6 = L.BatchNormalization(256)
            self.bnc7 = L.BatchNormalization(512)
            self.bnc8 = L.BatchNormalization(512)

            self.bnd8 = L.BatchNormalization(512)
            self.bnd7 = L.BatchNormalization(256)
            self.bnd6 = L.BatchNormalization(256)
            self.bnd5 = L.BatchNormalization(128)
            self.bnd4 = L.BatchNormalization(128)
            self.bnd3 = L.BatchNormalization(64)
            self.bnd2 = L.BatchNormalization(64)
            self.bnd1 = L.BatchNormalization(32)

    def calc(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(F.concat([e7, e8]))))
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        d6 = F.relu(self.bnd6(self.dc6(F.concat([e6, d7]))))
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        d4 = F.relu(self.bnd4(self.dc4(F.concat([e4, d5]))))
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        d2 = F.relu(self.bnd2(self.dc2(F.concat([e2, d3]))))
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        d0 = self.dc0(F.concat([e0, d1]))

        return d0

    def __call__(self, x, t):
        h = self.calc(x)
        loss = F.mean_absolute_error(h, t)
        chainer.report({'loss': loss}, self)
        return loss
