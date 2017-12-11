import chainer
import numpy as np
import nibabel as nib

import model as _model
import dataset as _dataset
from chainer import iterators

from argparse import ArgumentParser

feature_size = 1000

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--testImageIndex', default=0, type=int)
    parser.add_argument('--testBatchsize', default=128, type=int)
    parser.add_argument('--nsample', default=128)
    parser.add_argument('--model', nargs=1)
    parser.add_argument('--result', default='feature')
    parser.add_argument('--split_inter', default=True)
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    parser.add_argument('--output', default='grad.npz')
    args = parser.parse_args()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir,
                                                        split_inter=args.split_inter)
    _, test_dataset = all_dataset.get_subdatasets()
    test_itr = iterators.SerialIterator(dataset=test_dataset,
                                        batch_size=1,
                                        repeat=False, shuffle=False)

    target_img = test_dataset[args.testImageIndex]

    # preprocessing
    batch_size = args.testBatchsize
    batch = np.copy(np.broadcast_to(target_img,
                            [batch_size] + list(target_img.shape)))
    noise_level = 0.2
    sigma = noise_level / (np.max(target_img) - np.min(target_img))
    batch += sigma * np.random.randn(*batch.shape)

    mask = np.array(nib.load(args.mask).get_data(), dtype=np.float32)
    model = _model.ThreeDimensionalAutoEncoder(mask)

    chainer.serializers.load_npz(args.model[0], model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        batch = chainer.cuda.to_gpu(batch, args.gpu)

    x = chainer.Variable(batch)
    feature = model.extract(x)
    assert feature.shape == (batch_size, feature_size, 1, 1, 1)
    feature_coordinate = (0, 0, 0, 0)
    _feature = chainer.functions.sum(chainer.functions.get_item(feature, [Ellipsis] + list(feature_coordinate)))
    _feature.backward()
    grad = chainer.cuda.to_cpu(x.grad)
    with open(args.output, "wb") as f:
        np.savez(f, grad=grad)

    model.cleargrads()

if __name__ == '__main__':
    main()
