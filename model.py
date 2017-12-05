import chainer
import numpy as np

import chainer.functions as F
import chainer.links as L
import chainer.initializers as I


class ThreeDimensionalAutoEncoder(chainer.Chain):
    in_size = (91, 109, 91)

    # valid_range = [slice(10, 80, None), slice(11, 99, None), slice(3, 77, None)]

    def __init__(self, mask):
        super().__init__()

        self.mask = chainer.Variable(
            self.xp.array(mask != 0, dtype=self.xp.float32))
        assert tuple(mask.shape) == self.in_size

        self.idx_mask = self.mask.data.nonzero()
        assert isinstance(self.idx_mask, tuple)

        self.len_idx_mask = self.xp.count_nonzero(self.mask.data)

        initializer = I.LeCunNormal()
        with self.init_scope():
            self.linear1 = L.Linear(self.len_idx_mask, 1000)
            self.linear2 = L.Linear(1000, self.len_idx_mask)

    def to_cpu(self):
        super().to_cpu()
        self.mask.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.mask.to_gpu(device)

    def calc(self, x):
        _shape = list(x.shape)
        try:
            assert tuple(_shape[1:]) == (self.len_idx_mask,)
        except AssertionError:
            print("expected:{}, actual:{}"
                  .format((self.len_idx_mask,), tuple(_shape[1:])))
            exit()

        h = self.linear1(x)
        y = self.linear2(h)

        return y

    def __call__(self, x):
        assert tuple(x.shape[1:]) == self.in_size
        x_reshaped = x[[None].extend(self.idx_mask)]
        y_reshaped = self.calc(x_reshaped)

        loss = F.mean_absolute_error(y_reshaped, x_reshaped)
        chainer.report({'loss': loss}, self)
        return loss
