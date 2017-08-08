import chainer

import chainer.functions as F
import chainer.links as L


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (91, 109, 91)

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(3, 1, 16, (3, 5, 3), 2, 0)
            self.conv2 = L.ConvolutionND(3, 16, 32, (5, 7, 5), 2, 0)
            self.conv3 = L.ConvolutionND(3, 32, 64, (7, 8, 7), 2, 0)
            self.conv4 = L.ConvolutionND(3, 64, 128, (8, 9, 8), 1, 0)
            self.l1 = L.Linear(None, 150)
            self.bn1 = L.BatchNormalization(150)
            self.l2 = L.Linear(None, 20)
            self.bn2 = L.BatchNormalization(20)
            self.l3 = L.Linear(None, 150)
            self.bn3 = L.BatchNormalization(150)
            self.l4 = L.Linear(None, 128)
            self.bn4 = L.BatchNormalization(128)
            self.deconv4 = L.DeconvolutionND(3, 128, 64, (8, 9, 8), 1, 0)
            self.deconv3 = L.DeconvolutionND(3, 64, 32, (7, 8, 7), 2, 0)
            self.deconv2 = L.DeconvolutionND(3, 32, 16, (5, 7, 5), 2, 0)
            self.deconv1 = L.DeconvolutionND(3, 16, 1, (3, 5, 3), 2, 0)

    def calc(self, x):
        _shape = list(x.shape)
        _shape.insert(1, 1)

        c1 = F.tanh(self.conv1(F.reshape(x,tuple(_shape))))
        _shape = None
        c2 = F.tanh(self.conv2(c1))
        c3 = F.tanh(self.conv3(c2))
        c4 = F.tanh(self.conv4(c3))
        l1 = F.tanh(self.bn1(self.l1(c4)))
        l2 = F.tanh(self.bn2(self.l2(l1)))
        l3 = F.tanh(self.bn3(self.l3(l2)))
        l4 = F.tanh(self.bn4(self.l4(l3)))
        b4 = F.reshape(l4,(l4.shape[0], 128, 1, 1, 1))
        b3 = F.tanh(self.deconv4(b4))
        b2 = F.tanh(self.deconv3(b3))
        b1 = F.tanh(self.deconv2(b2))
        y = F.tanh(self.deconv1(b1))

        _shape = list(y.shape)
        _shape.pop(1)

        return F.reshape(y, tuple(_shape))

    def __call__(self, x, t):
        y = self.calc(x)
        loss = F.mean_absolute_error(y, t)
        chainer.report({'loss': loss}, self)
        return loss
