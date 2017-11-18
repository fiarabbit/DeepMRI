import chainer

import chainer.functions as F
import chainer.links as L


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (91, 109, 91) # = 902629

    def __init__(self, mask):
        assert tuple(mask.shape) == self.in_size
        super().__init__()
        self.mask = chainer.Variable(mask)
        with self.init_scope():
            self.conv1 = L.ConvolutionND(3, 1, 3, (7, 7, 7), 2, 0)
            # (3, 43, 52, 43) = 288444 = 32%
            self.bnc1 = L.BatchNormalization(3)
            self.conv2 = L.ConvolutionND(3, 3, 10, (5, 6, 5), 2, 0)
            # (10, 20, 24, 20) = 96000 = 33%
            self.bnc2 = L.BatchNormalization(10)
            self.conv3 = L.ConvolutionND(3, 10, 35, (4, 4, 4), 2, 0)
            # (35, 9, 11, 9) = 31185 = 32.5%
            self.bnc3 = L.BatchNormalization(35)
            self.deconv3 = L.DeconvolutionND(3, 35, 10, (4, 4, 4), 2, 0)
            # (10, 20, 24, 20)
            self.bnd3 = L.BatchNormalization(10)
            self.deconv2 = L.DeconvolutionND(3, 10, 3, (5, 6, 5), 2, 0)
            # (3, 43, 52, 32)
            self.bnd2 = L.BatchNormalization(3)
            self.deconv1 = L.DeconvolutionND(3, 3, 1, (7, 7, 7), 2, 0)
            # (1, 91, 109, 91)
            self.bnd1 = L.BatchNormalization(1)

    def to_cpu(self):
        super().to_cpu()
        self.mask.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask.to_gpu(device)

    def extract(self, x):
        _shape = list(x.shape)
        try:
            assert tuple(_shape[1:]) == self.in_size
        except AssertionError:
            print("expected:{}, actual:{}".format(self.in_size, tuple(_shape[1:])))
            exit()

        _shape.insert(1, 1)  # specify # of first channel

        c0 = F.reshape(x, tuple(_shape))
        c1 = F.relu(self.bnc1(self.conv1(c0)))
        c2 = F.relu(self.bnc2(self.conv2(c1)))
        return self.conv3(c2)

    def calc(self, x):
        _shape = list(x.shape)
        try:
            assert tuple(_shape[1:]) == self.in_size
        except AssertionError:
            print("expected:{}, actual:{}".format(self.in_size, tuple(_shape[1:])))
            exit()

        _shape.insert(1, 1)  # specify # of first channel

        c0 = F.reshape(x, tuple(_shape))
        c1 = F.relu(self.bnc1(self.conv1(c0)))
        c2 = F.relu(self.bnc2(self.conv2(c1)))
        c3 = F.relu(self.bnc3(self.conv3(c2)))
        b2 = F.relu(self.bnd3(self.deconv3(c3)))
        b1 = F.relu(self.bnd2(self.deconv2(b2)))
        y = F.relu(self.bnd1(self.deconv1(b1)))

        _shape = list(y.shape)
        _shape.pop(1)

        return F.reshape(y, tuple(_shape))

    def __call__(self, x):
        x_masked = F.scale(x, self.mask, axis=1)
        y = self.calc(x_masked)
        y_masked = F.scale(y, self.mask, axis=1)
        batch_size = y_masked.shape[0]
        loss = F.mean_absolute_error(y_masked, x_masked) \
            * y_masked.data.ravel().size \
            / (self.mask.data.sum() * batch_size)
        chainer.report({'loss': loss}, self)
        return loss
