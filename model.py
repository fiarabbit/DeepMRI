import chainer

import chainer.functions as F
import chainer.links as L


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (91, 109, 91) # = 902629 (effective number == mask == 150350)

    def __init__(self, mask):
        assert tuple(mask.shape) == self.in_size
        super().__init__()
        self.mask = chainer.Variable(mask)

    def to_cpu(self):
        super().to_cpu()
        self.mask.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask.to_gpu(device)

    def extract(self, x):
        _shape = list(x.shape)
        xp = chainer.cuda.get_array_module(x)
        return chainer.Variable(xp.zeros(_shape[0]))

    def calc(self, x):
        _shape = list(x.shape)
        try:
            assert tuple(_shape[1:]) == self.in_size
        except AssertionError:
            print("expected:{}, actual:{}".format(self.in_size, tuple(_shape[1:])))
            exit()
        xp = chainer.cuda.get_array_module(x)
        return chainer.Variable(xp.zeros(_shape))

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
