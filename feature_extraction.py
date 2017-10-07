import chainer
import numpy as np
import nibabel as nib

import model as _model
import dataset as _dataset
from chainer import iterators

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1)
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--testBatchsize', default=64)
    parser.add_argument('--model', nargs=1)
    parser.add_argument('--result', default='feature')
    parser.add_argument('--split',
                        choices=['inter', 'intra'], default='inter')
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    args = parser.parse_args()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir,
                                                        split_inter=args.split)
    train_dataset, test_dataset = all_dataset.get_subdatasets()
    test_itr = iterators.SerialIterator(dataset=test_dataset,
                                        batch_size=args.testBatchsize,
                                        repeat=False, shuffle=False)

    mask = np.array(nib.load(args.mask).get_data(), dtype=np.float32)
    model = _model.ThreeDimensionalAutoEncoder(mask)

    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    converter = chainer.dataset.convert.concat_examples

    stack = chainer.cuda.to_cpu(model.extract(converter(next(test_itr))).data)

    while True:
        try:
            _batch = next(test_itr)
            batch = converter(_batch)
            feature = chainer.cuda.to_cpu(model.extract(batch).data)
            stack = np.concatenate((stack, feature), 0)
        except StopIteration:
            break

    np.savez_compressed('feature.npz', data=stack)


if __name__ == '__main__':
    main()
