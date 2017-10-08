import chainer
import numpy as np
import nibabel as nib

import model as _model
import dataset as _dataset
from chainer import iterators

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
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

    chainer.serializers.load_npz(args.model[0], model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    def converter(_batch):
        return chainer.dataset.concat_examples(_batch, device=args.gpu)

    tmp = chainer.cuda.to_cpu(model.extract(converter(next(test_itr))).data)
    stack = np.zeros([len(test_dataset)] + list(tmp.shape[1:]),
                     dtype=np.float32)
    test_itr.reset()

    i = 0
    while True:
        try:
            start_idx = i * args.testBatchsize
            end_idx = np.min([(i + 1) * args.testBatchsize, len(test_dataset)])
            print("{}...{}/{}".format(start_idx, end_idx, len(test_dataset)))
            _batch = next(test_itr)
            batch = converter(_batch)
            feature = chainer.cuda.to_cpu(model.extract(batch).data)
            stack[start_idx:end_idx, :] = feature
            i += 1
        except StopIteration:
            break

    np.savez_compressed('feature.npz', data=stack)


if __name__ == '__main__':
    main()
