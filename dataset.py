import chainer
from chainer import dataset

from chainer.dataset import DatasetMixin

class AutoEncoderTrainDataset(DatasetMixin): # DatasetMixin supports default slicing
    """ :meth:get_example returns numpy.ndarray of image

    :ivar insize: size of image (i.e. 1st layer)
    :ivar imgRoot: root directory of images
    :ivar 

    """

    def __init__(self, imgRoot, model):
        """

        :param str imgRoot: root directory of images
        :param chainer.chainer.Chain model: model to be learned

        """

        self.imgRoot = imgRoot
        self.insize = model.insize

        pass

    def __len__(self):
        pass

    def get_example(self, i):
        pass