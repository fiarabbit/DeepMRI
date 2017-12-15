import chainer
import numpy as np
import nibabel as nib

import model as _model
import dataset as _dataset
from chainer import iterators

from argparse import ArgumentParser
from os.path import join
from time import time

feature_size = 1000

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--testBatchsize', default=128, type=int)
    parser.add_argument('--nsample', default=128)
    parser.add_argument('--model', nargs=1)
    parser.add_argument('--result', default='feature')
    parser.add_argument('--split_inter', default=True)
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    parser.add_argument('--output', default='./grad')
    parser.add_argument('--feature', default=0, type=int)
    args = parser.parse_args()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir,
                                                        split_inter=args.split_inter)
    with open("parameters.txt", "a") as f:
        print("subjects:{}".format(all_dataset.subjects), file=f)
        print("testBatchsize:{}".format(args.testBatchsize), file=f)
        print("model:{}".format(args.model), file=f)

    _, test_dataset = all_dataset.get_subdatasets()

    s = time()
    stack_ave_grad = []
    with open("log.txt", "a") as f:
        print("i,sigma", file=f)
        for i, test_image_index in enumerate(range(len(test_dataset))):
            target_img = test_dataset[test_image_index]

            # preprocessing
            batch_size = args.testBatchsize
            batch = np.copy(np.broadcast_to(target_img,
                                    [batch_size] + list(target_img.shape)))
            noise_level = 0.2
            sigma = noise_level / (np.max(target_img) - np.min(target_img))
            print("{},{}".format(i, sigma), file=f)
            n = time()
            print("processing {}/{} image {:.2f}s elapsed".format(i+1, len(test_dataset), n-s))
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
            feature_coordinate = (args.feature, 0, 0, 0)
            _feature = chainer.functions.sum(chainer.functions.get_item(feature, [Ellipsis] + list(feature_coordinate)))
            _feature.backward()
            grad = chainer.cuda.to_cpu(x.grad)
            stack_ave_grad.append(np.copy(mask * np.abs(grad).mean(axis=0)))
            # with open(join(args.output, "grad_{}.npz".format(i)), "wb") as _f:
            #     np.savez_compressed(_f, grad=grad)
            if i % 150 == 149:
                ave_grad = np.stack(stack_ave_grad)
                with open(join(args.output, "grad_subject{}_feature{}.npz".format(i//150, args.feature)), "wb") as f_subject:
                    np.savez_compressed(f_subject, data=ave_grad)
                    print("saving {}th subject".format(i//150 + 1))
                stack_ave_grad=[]
            model.cleargrads()


if __name__ == '__main__':
    main()
