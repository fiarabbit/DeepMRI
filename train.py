#!/usr/bin/env python
"""module for training

Program to set up a trainer, and execute trainer.run().

trainer is composed of
- Dataset
- Iterator
- Chain ("model" in other deep learning architectures)
- Optimizer
- Updater
- extensions

These units are arranged as follows:

trainer - updater     - iterator - dataset
       |- extensions |- optimizer - chain (model)

Typically, users only have to implement chain and dataset.
Users also have to choose optimizer from libraries.

Usually, the default trainer, updater and iterator works well,
so Users do not have to implement it.

"""
import chainer.training
from chainer import datasets, iterators, Chain, optimizers
from chainer.training import updater as updaters
from chainer.training import extensions, triggers

import numpy

import model

if __name__ == '__main__':
    """This module should be run via command line."""

    dataset = datasets.ImageDataset(paths='./images/imagelist.txt',
                                    root='./images')
    iterator = iterators.SerialIterator(dataset=dataset, batch_size=16,
                                        repeat=True, shuffle=True)
    model = model.UNET128()
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    updater = updaters.StandardUpdater(iterator=iterator, optimizer=optimizer,
                                       device=None)
    trainer = chainer.training.Trainer(updater=updater,
                                       stop_trigger=(120000, 'iteration'),
                                       out='result')
    # if you use SGD, following extension has to be set
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=0.001),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration')
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
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'lr']),
        trigger=log_interval
    )

    trainer.run()
