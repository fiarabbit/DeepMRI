import nibabel as nib
import numpy as np
from os import listdir
from os.path import join

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.ioff()
import time
import dataset as _dataset
from chainer import iterators
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
import cupy
import numpy
import matplotlib.animation as ani
from scipy.spatial.distance import cosine


def individual_mean(gpu=0):
    if gpu >= 0:
        xp = cupy
    else:
        xp = numpy

    datasetdir = '/data/timeseries'
    split_inter = True
    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(datasetdir,
                                                        split_inter=split_inter)
    train_dataset, test_dataset = all_dataset.get_subdatasets()
    mask = np.array(nib.load('/data/mask/average_optthr.nii').get_data())
    test_itr = iterators.SerialIterator(dataset=test_dataset,
                                        batch_size=150,
                                        repeat=False, shuffle=False)

    stack_cossim = []
    idx_mask = mask.nonzero()
    i = 0
    while True:
        try:
            print("{}/{}".format(i * 64, len(test_dataset)))
            batch = concat_examples(next(test_itr))
            input_batch_data = batch[[Ellipsis] + list(idx_mask)]
            output_batch_data = input_batch_data.mean(axis=0)
            stack_cossim.append(
                np.array(
                    [1 - cosine(input_batch_data[j, :], output_batch_data)
                     for j in
                     range(input_batch_data.shape[0])]
                )
            )
            i += 1
        except StopIteration:
            break

    stack_cossim = np.hstack(stack_cossim)
    with open("i_stack_cossim.npz", "wb") as f:
        np.savez(f, data=stack_cossim)

def global_mean(gpu=0):
    if gpu >= 0:
        xp = cupy
    else:
        xp = numpy

    datasetdir = '/data/timeseries'
    split_inter = True
    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(datasetdir,
                                                        split_inter=split_inter)
    train_dataset, test_dataset = all_dataset.get_subdatasets()
    mask = np.array(nib.load('/data/mask/average_optthr.nii').get_data())
    train_itr = iterators.SerialIterator(dataset=train_dataset,
                                         batch_size=64,
                                         repeat=False, shuffle=False)
    test_itr = iterators.SerialIterator(dataset=test_dataset,
                                        batch_size=64,
                                        repeat=False, shuffle=False)
    # stack = []
    # i = 0
    # while True:
    #     try:
    #         print("{}/{}".format(i*64, len(train_dataset)))
    #         batch = concat_examples(next(train_itr), gpu)
    #         stack.append(to_cpu(batch.sum(axis=0)))
    #         i += 1
    #     except StopIteration:
    #         break
    #
    # g_mean = np.stack(stack).sum(axis=0) / len(train_dataset)
    # with open("g_mean.npz", "wb") as f:
    #     np.savez(f, data=g_mean)

    with open("g_mean.npz", "rb") as f:
        g_mean = np.load(f)["data"]

    assert isinstance(g_mean, np.ndarray)

    stack_cossim = []
    idx_mask = mask.nonzero()
    i = 0
    while True:
        try:
            print("{}/{}".format(i*64, len(test_dataset)))
            batch = concat_examples(next(test_itr))
            input_batch_data = batch[[Ellipsis] + list(idx_mask)]
            output_batch_data = g_mean[list(idx_mask)]
            stack_cossim.append(
                np.array(
                    [1 - cosine(input_batch_data[j, :], output_batch_data) for j in
                     range(input_batch_data.shape[0])]
                )
            )
            i+=1
        except StopIteration:
            break

    stack_cossim = np.hstack(stack_cossim)
    with open("g_stack_cossim.npz", "wb") as f:
        np.savez(f, data=stack_cossim)

def check_naive():
    datasetdir = '/data/timeseries'
    split_inter = True
    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(datasetdir,
                                                        split_inter=split_inter)
    train_dataset, test_dataset = all_dataset.get_subdatasets()

    mask = np.array(nib.load('/data/mask/average_optthr.nii').get_data())
    mask[mask != 0] = 1

    loss = np.zeros((len(test_dataset),))
    for i, d in enumerate(test_dataset):
        d = d * mask
        loss[i] = np.abs(d.ravel()).mean() * d.size / len(
            mask.ravel().nonzero()[0])

    print(loss.mean())  # 0.283901693217


def calcoutpsize(insize, kernel, stride, padding):
    return (input + 2 * padding - kernel) / stride + 1


def anim():
    x = np.arange(6)
    y = np.arange(5)
    z = x * y[:, np.newaxis]

    for i in range(5):
        if i == 0:
            p = plt.imshow(z)
            # fig = plt.gcf()
            plt.clim()  # clamp the color limits
            plt.title("Boring slide show")
        else:
            z = z + 2
            p.set_data(z)

        print("step", i)
        plt.pause(0.5)


def mask():
    # h = nib.load('average_COBRE148subjects.nii')
    h = nib.load('subject06204_swaurest.nii')
    d = h.dataobj
    frame = 0
    f = d[:, :, :, frame]

    # f = d
    class AnimationFunc:
        s = 0
        im = None

        def __init__(self, img_3d):
            self.img_3d = img_3d

        def __call__(self):
            img_2d = self.img_3d[:, :, self.s]
            print(img_2d.max())
            if self.s == 0:
                self.im = plt.imshow(img_2d)
                self.im.set_clim(0, 2000)
            elif self.s > 0:
                self.im.set_data(img_2d)
            plt.pause(0.1)
            self.s += 1
            print(self.s)

    a = AnimationFunc(f)
    while True:
        a()


def calcHist():
    datasetdir = '/data/timeseries'
    files = listdir(datasetdir)
    len_files = len(files)
    len_frames = 150
    h = np.histogram(np.hstack(
        [nib.load(join(datasetdir, files[l])).dataobj for l in range(5)]),
        bins=100)
    b = h[1].copy()
    np.savez('bin.npz', b=b)
    h_list = []
    for l in range(len_files):
        h_list.append(
            np.histogram(nib.load(join(datasetdir, files[l])).dataobj, bins=b)[
                0])
        np.savez('histogram.npz', h_list=h_list)
        print(h_list)


def test_calcHist():
    datasetdir = '/data/timeseries'
    files = listdir(datasetdir)
    len_files = len(files)
    len_frames = 150
    ret = np.zeros((len_files,))
    for l in range(len_files):
        ret[l] = nib.load(join(datasetdir, files[l])).get_data().mean()
        print(ret[l])
    np.savez('mean_list', mean_list=ret)


def plotHist():
    h_list = np.load('histogram.npz')['h_list']
    b = np.load('bin.npz')['b']
    s = h_list.sum(axis=0)
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)

    class Anime():
        k = 0

        def __call__(self, i):
            if self.k == 0:
                self.p = ax.bar(b[0:-1], h_list[0, :])
            elif self.k < h_list.shape[0]:
                for i, n in enumerate(self.p):
                    n.set_height(h_list[self.k, i])
            self.k += 1

    a = Anime()
    anim = ani.FuncAnimation(fig, a)
    plt.show()


def plotMeanHist():
    meanArr = np.load('mean_list.npz')['mean_list']
    print(meanArr.mean())
    plt.hist(meanArr)
    plt.show()


if __name__ == '__main__':
    individual_mean()
