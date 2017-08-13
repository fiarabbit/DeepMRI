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

import model as _model
import dataset as _dataset

import os

print(os.getcwd())


def main():
    """ called if __name__==='__main__' """
    parser = ArgumentParser()
    parser.add_argument('--datasetdir', default='/data/timeseries')
    parser.add_argument('--split',
                        choices=['inter', 'intra'], default='inter')
    parser.add_argument('--split_ratio', type=tuple, default=(4, 1))
    # parser.add_argument('--traindir', default='./data/timeseries/train')
    # parser.add_argument('--testdir', default='./data/timeseries/test')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--output', default='result')
    args = parser.parse_args()

    model = _model.ThreeDimensionalAutoEncoder()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.datasetdir,
                                                        split_inter=args.split)
    train_dataset, test_dataset = all_dataset.get_subdatasets()

    # train_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.traindir)
    # test_dataset = _dataset.TimeSeriesAutoEncoderDataset(args.testdir)
    train_iterator = iterators.SerialIterator(dataset=train_dataset,
                                              batch_size=args.batchsize,
                                              repeat=True,
                                              shuffle=True)
    test_iterator = iterators.SerialIterator(dataset=test_dataset,
                                             batch_size=args.batchsize,
                                             repeat=False, shuffle=False)
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    # optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(0.0005))

    updater = updaters.StandardUpdater(iterator=train_iterator,
                                       optimizer=optimizer,
                                       device=args.gpu)
    trainer = chainer.training.Trainer(updater=updater,
                                       stop_trigger=(120000, 'iteration'),
                                       out='result')
    # if you use SGD, following extension has to be set
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=0.001),
        trigger=triggers.ManualScheduleTrigger([220, 280], 'epoch')
    )

    snapshot_interval = (1000, 'iteration')
    log_interval = (10, 'iteration')

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'),
        trigger=log_interval
    )
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.Evaluator(test_iterator, model, device=args.gpu),
                   trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'lr']),
        trigger=log_interval
    )

    trainer.run()


if __name__ == '__main__':
    main()
