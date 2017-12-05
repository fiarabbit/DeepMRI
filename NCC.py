import chainer
import numpy as np
import nibabel as nib

import model as _model
import dataset as _dataset
from chainer import iterators

from argparse import ArgumentParser
from chainer import functions as F
from scipy.spatial.distance import cosine

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--testBatchsize', default=32, type=int)
    parser.add_argument('--model', nargs=1)
    parser.add_argument('--result', default='feature')
    parser.add_argument('--split_inter', default=True)
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    args = parser.parse_args()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir,
                                                        split_inter=args.split_inter)
    _, test_dataset = all_dataset.get_subdatasets()
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

    i = 0
    while True:
        try:
            start_idx = i * args.testBatchsize
            end_idx = np.min([(i + 1) * args.testBatchsize, len(test_dataset)])
            print("{}...{}/{}".format(start_idx, end_idx, len(test_dataset)))
            _batch = next(test_itr)
            batch = converter(_batch)
            input_batch = F.scale(batch, model.mask, axis=1)
            output_batch = model.calc(input_batch)
            output_batch = F.scale(output_batch, model.mask, axis=1)
            try:
                xp
            except NameError:
                xp = chainer.cuda.get_array_module(output_batch)

            abs_loss = xp.abs(output_batch.data - input_batch.data)
            loss_batch = xp.mean(abs_loss, axis=tuple(range(1, len(output_batch.shape))))

            try:
                stack_loss
                stack_cossim
            except NameError:
                stack_loss = xp.zeros([len(test_dataset)])
                stack_cossim = xp.zeros([len(test_dataset)])

            print(stack_loss[start_idx:end_idx])
            print(loss_batch)

            stack_loss[start_idx:end_idx] = loss_batch
            stack_cossim[start_idx:end_idx] = np.array([1-cosine(input_batch.data[j,], output_batch.data[j,]) for j in range(input_batch.shape[0])])
            i += 1
        except StopIteration:
            break

    stack_loss = chainer.cuda.to_cpu(stack_loss)
    stack_cossim = chainer.cuda.to_cpu(stack_cossim)

    with open("stack_loss.npz", "wb") as f:
        np.savez(f,data=stack_loss)

    with open("stack_cossim.npz", "wb") as f:
        np.savez(f, data=stack_cossim)

if __name__ == '__main__':
    main()
