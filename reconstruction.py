import chainer
import numpy as np
import nibabel as nib

import model as _model
import dataset as _dataset
from chainer import iterators

from argparse import ArgumentParser
from os.path import join


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--testBatchsize', default=150)
    parser.add_argument('--model', nargs=1)
    parser.add_argument('--output', default='./reconstruction')
    parser.add_argument('--split_inter', default=True)
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    args = parser.parse_args()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir,
                                                        split_inter=args.split_inter)
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

    y_stack = np.zeros([150, 91, 109, 91], dtype=np.float32)
    diff_stack = np.zeros([150, 91, 109, 91], dtype=np.float32)
    test_itr.reset()

    i_sub = 0
    while True:
        try:
            print("{}/{}".format(i_sub, len(test_dataset)))
            _batch = next(test_itr)
            batch = converter(_batch)
            batch_masked = chainer.functions.scale(batch, model.mask, axis=1)
            y = model.calc(batch_masked)
            y_masked = chainer.functions.scale(y, model.mask, axis=1)
            y_stack[:, :, :, :] = chainer.cuda.to_cpu(y_masked.data)
            diff = y - y_masked
            diff_stack[:, :, :, :] = chainer.cuda.to_cpu(diff.data)
            np.savez_compressed(
                join(args.output, 'reconstruction_subject{}.npz').format(
                    i_sub), y=y_stack, diff=diff_stack)
            i_sub += 1
        except StopIteration:
            break


if __name__ == '__main__':
    main()
