import chainer

import chainer.functions as F
import chainer.links as L


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (91, 109, 91)

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(3, 1, 16, (3, 5, 3), 2, 0)
            self.bnc1 = L.BatchNormalization(16)
            self.conv2 = L.ConvolutionND(3, 16, 32, (5, 7, 5), 2, 0)
            self.bnc2 = L.BatchNormalization(32)
            self.conv3 = L.ConvolutionND(3, 32, 64, (7, 8, 7), 2, 0)
            self.bnc3 = L.BatchNormalization(64)
            self.conv4 = L.ConvolutionND(3, 64, 20, (8, 9, 8), 1, 0)
            self.bnc4 = L.BatchNormalization(20)
            self.deconv4 = L.DeconvolutionND(3, 20, 64, (8, 9, 8), 1, 0)
            self.bnd4 = L.BatchNormalization(64)
            self.deconv3 = L.DeconvolutionND(3, 64, 32, (7, 8, 7), 2, 0)
            self.bnd3 = L.BatchNormalization(32)
            self.deconv2 = L.DeconvolutionND(3, 32, 16, (5, 7, 5), 2, 0)
            self.bnd2 = L.BatchNormalization(16)
            self.deconv1 = L.DeconvolutionND(3, 16, 1, (3, 5, 3), 2, 0)

    def calc(self, x):
        _shape = list(x.shape)
        _shape.insert(1, 1)  # specify # of first channel

        c0 = F.reshape(x, tuple(_shape))
        c1 = F.relu(self.bnc1(self.conv1(c0)))
        c2 = F.relu(self.bnc2(self.conv2(c1)))
        c3 = F.relu(self.bnc3(self.conv3(c2)))
        c4 = F.relu(self.bnc4(self.conv4(c3)))
        b3 = F.relu(self.bnd4(self.deconv4(c4)))
        b2 = F.relu(self.bnd3(self.deconv3(b3)))
        b1 = F.relu(self.bnd2(self.deconv2(b2)))
        y = F.relu(self.deconv1(b1))

        _shape = list(y.shape)
        _shape.pop(1)

        return F.reshape(y, tuple(_shape))

    def __call__(self, x, t):
        y = self.calc(x)
        loss = F.mean_absolute_error(y, t)
        chainer.report({'loss': loss}, self)
        return loss
