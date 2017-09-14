import chainer
import numpy as np
import nibabel as nib

import model as _model

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1)
    parser.add_argument('--model', nargs=1)
    parser.add_argument('--result', default='feature')
    args = parser.parse_args()

    mask = np.array(nib.load(args.mask).get_data(), dtype=np.float32)
    model = _model.ThreeDimensionalAutoEncoder(mask)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    pass

