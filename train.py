"""module for training

Program to set up a :class:`chainer.training.Trainer`,
and execute :meth:`~chainer.training.Trainer.run`.

:class:`~chainer.training.trainer.Trainer` contains::

- Dataset (chainer.dataset.DatasetMixin)
- Iterator (chainer.dataset.Iterator)
- Chain (chainer.Chain ["model" in other deep learning architectures])
- Optimizer (chainer.optimizer.Optimizer)
- Updater (chainer.training.updater.Updater)
- Extensions (chainer.training.extension.Extension)

These units are arranged as follows:


::

    Trainer
    ├── Updater
    │   ├── Iterator
    │   │   └── Dataset
    │   └── Optimizer
    │       └── Chain(model)
    └── extensions

Typically, users only have to implement :class:`~chainer.Chain`
and :class:`Dataset <chainer.dataset.DatasetMixin>`.
Users also have to choose :class:`~chainer.optimizer.Optimizer`
from :mod:`chainer.optimizers`.

Usually the default :class:`~chainer.training.Trainer`,
:class:`Updater <~chainer.training.StandardUpdater>`,
and :class:`Iterator <~chainer.iterators.SerialIterator>` works well,
so users do not have to implement it.

"""

from argparse import ArgumentParser

import chainer.training
from chainer import iterators, optimizers
from chainer.optimizer import WeightDecay
from chainer.training import updater as updaters
from chainer.training import extensions, triggers
from chainer.serializers import load_npz

import nibabel as nib
import numpy as np

import model as _model
import dataset as _dataset

import os
import warnings

print(os.getcwd())


def main():
    """ called if __name__==='__main__' """
    parser = ArgumentParser()
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--split_inter', default=True)
    parser.add_argument('--split_ratio', type=tuple, default=(4, 1))
    # parser.add_argument('--traindir', default='./data/timeseries/train')
    # parser.add_argument('--testdir', default='./data/timeseries/test')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--testBatchsize', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--output', default='result')
    parser.add_argument('--resumeFrom')
    parser.add_argument('--exponentialShift', default=1, type=float)
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    args = parser.parse_args()

    mask = np.array(nib.load(args.mask).get_data(), dtype=np.float32)
    try:
        assert (1 == mask[mask.nonzero()]).all()
    except AssertionError:
        warnings.warn("Non-bool mask Warning")
        print("converting to boolean...")
        mask[mask.nonzero()] = 1
        mask = mask.astype(np.float32)

        assert (1 == mask[mask.nonzero()]).all()

    model = _model.ThreeDimensionalAutoEncoder(mask=mask)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir, mask,
                                                        split_inter=args.split_inter)
    train_dataset, test_dataset = all_dataset.get_subdatasets()

    # train_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.traindir)
    # test_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.testdir)
    # train_iterator = iterators.SerialIterator(dataset=train_dataset,
    #                                           batch_size=args.batchsize,
    #                                           repeat=True,
    #                                           shuffle=True)
    # test_iterator = iterators.SerialIterator(dataset=test_dataset,
    #                                          batch_size=args.testBatchsize,
    #                                          repeat=False, shuffle=False)
    train_iter = iterators.MultiprocessIterator(dataset=train_dataset,
                                                batch_size=args.batchsize,
                                                repeat=True,
                                                shuffle=True)
    test_iter = iterators.MultiprocessIterator(dataset=test_dataset,
                                               batch_size=args.testBatchsize,
                                               repeat=False, shuffle=False)
    optimizer = optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    # optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(0.0005))

    updater = updaters.StandardUpdater(iterator=train_iter,
                                       optimizer=optimizer,
                                       device=args.gpu)
    trainer = chainer.training.Trainer(updater=updater,
                                       stop_trigger=(30000, 'iteration'),
                                       out='result')
    model_interval = (100, 'iteration')
    snapshot_interval = (1000, 'iteration')
    log_interval = (10, 'iteration')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'),
        trigger=model_interval
    )
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    trainer.extend(evaluator, trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'lr',
         'elapsed_time']),
        trigger=log_interval
    )
    if args.resumeFrom is not None:
        load_npz(args.resumeFrom, trainer)
        optimizer.lr = optimizer.lr * args.exponentialShift

    # # if you use SGD, following extension has to be set
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=0.1),
        trigger=(20, 'epoch'))

    trainer.run()


if __name__ == '__main__':
    main()
