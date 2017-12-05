import chainer
import numpy as np

import chainer.functions as F
import chainer.links as L
import chainer.initializers as I


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (150350,)

    def __init__(self):
        super().__init__()

        initializer = I.LeCunNormal()
        with self.init_scope():
            self.linear1 = L.Linear(self.in_size[0], 1000)
            self.linear2 = L.Linear(1000, self.in_size[0])

    def to_cpu(self):
        super().to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)

    def calc(self, x):
        _shape = list(x.shape)
        try:
            assert tuple(_shape[1:]) == self.in_size
        except AssertionError:
            print("expected:{}, actual:{}"
                  .format(self.in_size, tuple(_shape[1:])))
            exit()

        h = self.linear1(x)
        y = self.linear2(h)

        return y

    def __call__(self, x):
        y = self.calc(x)
        loss = F.mean_absolute_error(y, x)
        chainer.report({'loss': loss}, self)
        return loss
