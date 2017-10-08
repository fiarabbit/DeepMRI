import chainer

import chainer.functions as F
import chainer.links as L


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (91, 109, 91)

    def __init__(self, mask):
        assert tuple(mask.shape) == self.in_size
        super().__init__()
        self.mask = chainer.Variable(mask)
        with self.init_scope():
            self.conv1 = L.ConvolutionND(3, 1, 3, (7, 7, 7), 2, 0)
            self.bnc1 = L.BatchNormalization(3)
            self.conv2 = L.ConvolutionND(3, 3, 10, (5, 6, 5), 2, 0)
            self.bnc2 = L.BatchNormalization(10)
            self.conv3 = L.ConvolutionND(3, 10, 35, (4, 4, 4), 2, 0)
            self.bnc3 = L.BatchNormalization(35)
            self.conv4 = L.ConvolutionND(3, 35, 125, (3, 3, 3), 2, 0)
            self.bnc4 = L.BatchNormalization(125)
            self.conv5 = L.ConvolutionND(3, 125, 270, (3, 3, 3), 1, 0)
            self.bnc5 = L.BatchNormalization(270)
            self.conv6 = L.ConvolutionND(3, 270, 1000, (2, 3, 2), 1, 0)
            self.bnc6 = L.BatchNormalization(1000)
            self.l1 = L.Linear(None, 300)
            self.bnl1 = L.BatchNormalization(300)
            self.l2 = L.Linear(None, 100)
            self.bnl2 = L.BatchNormalization(100)
            self.l3 = L.Linear(None, 20)
            self.bnl3 = L.BatchNormalization(20)
            self.l4 = L.Linear(None, 100)
            self.bnl4 = L.BatchNormalization(100)
            self.l5 = L.Linear(None, 300)
            self.bnl5 = L.BatchNormalization(300)
            self.l6 = L.Linear(None, 1000)
            self.bnl6 = L.BatchNormalization(1000)
            self.deconv6 = L.DeconvolutionND(3, 1000, 270, (2, 3, 2), 1, 0)
            self.bnd6 = L.BatchNormalization(270)
            self.deconv5 = L.DeconvolutionND(3, 270, 125, (3, 3, 3), 1, 0)
            self.bnd5 = L.BatchNormalization(125)
            self.deconv4 = L.DeconvolutionND(3, 125, 35, (3, 3, 3), 2, 0)
            self.bnd4 = L.BatchNormalization(35)
            self.deconv3 = L.DeconvolutionND(3, 35, 10, (4, 4, 4), 2, 0)
            self.bnd3 = L.BatchNormalization(10)
            self.deconv2 = L.DeconvolutionND(3, 10, 3, (5, 6, 5), 2, 0)
            self.bnd2 = L.BatchNormalization(3)
            self.deconv1 = L.DeconvolutionND(3, 3, 1, (7, 7, 7), 2, 0)
            self.bnd1 = L.BatchNormalization(1)

    def to_cpu(self):
        super().to_cpu()
        self.mask.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask.to_gpu(device)

    def extract(self, x):
        _shape = list(x.shape)
        assert tuple(_shape[1:]) == self.in_size

        _shape.insert(1, 1) # specify # of first channel

        c0 = F.reshape(x, tuple(_shape))
        c1 = F.relu(self.bnc1(self.conv1(c0)))
        c2 = F.relu(self.bnc2(self.conv2(c1)))
        c3 = F.relu(self.bnc3(self.conv3(c2)))
        c4 = F.relu(self.bnc4(self.conv4(c3)))
        c5 = F.relu(self.bnc5(self.conv5(c4)))
        c6 = F.relu(self.bnc6(self.conv6(c5)))
        l1 = F.relu(self.bnl1(self.l1(c6)))
        l2 = F.relu(self.bnl2(self.l2(l1)))
        return self.l3(l2)

    def calc(self, x):
        _shape = list(x.shape)
        assert tuple(_shape[1:]) == self.in_size

        _shape.insert(1, 1)  # specify # of first channel

        c0 = F.reshape(x, tuple(_shape))
        c1 = F.relu(self.bnc1(self.conv1(c0)))
        c2 = F.relu(self.bnc2(self.conv2(c1)))
        c3 = F.relu(self.bnc3(self.conv3(c2)))
        c4 = F.relu(self.bnc4(self.conv4(c3)))
        c5 = F.relu(self.bnc5(self.conv5(c4)))
        c6 = F.relu(self.bnc6(self.conv6(c5)))
        l1 = F.relu(self.bnl1(self.l1(c6)))
        l2 = F.relu(self.bnl2(self.l2(l1)))
        l3 = F.relu(self.bnl3(self.l3(l2)))
        l4 = F.relu(self.bnl4(self.l4(l3)))
        l5 = F.relu(self.bnl5(self.l5(l4)))
        l6 = F.relu(self.bnl6(self.l6(l5)))
        b6 = F.reshape(l6, list(l6.shape) + [1, 1, 1])
        b5 = F.relu(self.bnd6(self.deconv6(b6)))
        b4 = F.relu(self.bnd5(self.deconv5(b5)))
        b3 = F.relu(self.bnd4(self.deconv4(b4)))
        b2 = F.relu(self.bnd3(self.deconv3(b3)))
        b1 = F.relu(self.bnd2(self.deconv2(b2)))
        y = F.relu(self.bnd1(self.deconv1(b1)))

        _shape = list(y.shape)
        _shape.pop(1)

        return F.reshape(y, tuple(_shape))

    def __call__(self, x):
        x_masked = F.scale(x, self.mask, axis=1)
        y = self.calc(x_masked)
        y_masked = F.scale(y, self.mask, axis=1)
        loss = F.mean_absolute_error(y_masked, x_masked) \
            * y_masked.data.ravel().size / self.mask.data.sum()
        chainer.report({'loss': loss}, self)
        return loss
