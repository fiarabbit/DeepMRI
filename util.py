import nibabel as nib
import numpy as np
from os import listdir
from os.path import join
from nibabel.processing import resample_from_to

import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.ioff()
import time
import dataset as _dataset

import matplotlib.animation as ani
import json

def imshow_change_data():
    root_dir = 'C:/Users/hashimoto/PycharmProjects/chainer2python3'
    mask_path = join(root_dir, 'average_optthr.nii')
    base_path = join(root_dir, 'average_COBRE148subjects.nii')

    mask = nib.load(mask_path).get_data()
    base = nib.load(base_path).get_data()




def grad_correlation():
    root_dir_d =  '/efs/replication_1000channel/DeepMRI/grad/'
    root_dir = '/data/mask'
    
    mask_path = join(root_dir, 'average_optthr.nii')
    base_path = join(root_dir, 'average_COBRE148subjects.nii')

    mask = nib.load(mask_path).get_data()
    base = nib.load(base_path).get_data()

    assert isinstance(mask, np.ndarray)
    assert isinstance(base, np.ndarray)
    assert mask.shape == (91, 109, 91)
    assert base.shape == (91, 109, 91)
    
    tmp_corr = []
    for i_subject in range(0,29,1):
        file_path = join(root_dir_d, 'grad_subject{}.npz'.format(i_subject))
        with open(file_path, "rb") as f:
            d = np.load(f)["data"]
            assert isinstance(d, np.ndarray)
            try:
                assert d.shape == (150, 91, 109, 91)
            except AssertionError:
                print(d.shape)
                exit()
            # d = np.arange(91*109*91*150).reshape([91, 109, 91, 150])
            d = np.moveaxis(d, 0, -1)
            try:
                assert d.shape == (91, 109, 91, 150)
            except AssertionError:
                print(d.shape)
                exit()
            d_valid = d[mask.nonzero()]
            assert d_valid.shape == (150350, 150)
            
            r = np.zeros([150, 150])
            for sample_1 in range(150):
                grad_1 = d_valid[:, sample_1]
                for sample_2 in range(sample_1):
                    grad_2 = d_valid[:, sample_2]
                    c = np.cov(np.vstack([grad_1, grad_2]))
                    r[sample_1,sample_2] = c[0,1]/np.sqrt(c[0,0]*c[1,1])
            tmp_corr.append(r.sum()/np.count_nonzero(r))
    with open("tmpcorr.npz","wb") as _f:
        np.savez_compressed(_f, data=tmp_corr)
    print(tmp_corr.mean())
            
    # plot
    exit()
    z_list = [33, 38, 43, 48, 53, 58, 63, 68]

    for z in z_list:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(base[:, :, z], cmap="gray")
        ax.imshow(d[:, :, z], cmap="hot", alpha=0.6)
        ax.get_images().set_clim(0, 1)
        plt.show()


def grad_anim():
    root_dir = 'C:/Users/hashimoto/PycharmProjects/chainer2python3'
    file_path = join(root_dir, 'grad.npz')
    mask_path = join(root_dir, 'average_optthr.nii')
    base_path = join(root_dir, 'average_COBRE148subjects.nii')

    mask = nib.load(mask_path).get_data()
    base = nib.load(base_path).get_data()
    print(base.shape)
    with open(file_path, "rb") as f:
        data = np.load(f)["grad"]

    m = mask * np.abs(data).mean(axis=0)

    for i in range(33, 73, 5):
        s = m[:, :, i]
        if i == 33:
            fig, axes = plt.subplots(1, 1)
            if not hasattr(axes, '__getitem__'):
                axes = [axes]
        axes[0].imshow(base[:, :, i], cmap='gray')
        axes[0].imshow(s, cmap='hot', alpha=0.6)
        im = axes[0].get_images()
        im[0].set_clim(0, 1)
        plt.pause(1)

def plot_loss():
    root_dir = 'C:/Users/hashimoto/PycharmProjects/chainer2python3'
    file_path = join(root_dir, 'log')
    with open(file_path, "r") as f:
        data_list = json.load(f)

    print(data_list[0].keys())

    main_loss = []
    validation_loss = []
    iteration = []
    for i, item in enumerate(data_list):
        main_loss.append(item['main/loss'])
        validation_loss.append(item['validation/main/loss'])
        iteration.append(item['iteration'])

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 3)
    ax.plot(iteration[2:], main_loss[2:])
    ax.plot(iteration[2:], validation_loss[2:])
    plt.show()

def feature_analysis():
    root_dir = 'C:/Users/hashimoto/PycharmProjects/chainer2python3'
    file_path = join(root_dir, 'feature.npz')
    mask_path = join(root_dir, 'average_optthr.nii')

    with open(file_path, "rb") as f:
        feature = np.load(f)["data"].squeeze()
        print(feature.shape)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 3)
    for s in range(0, 3, 1):
        d_s = feature[150 * s:150 * (s + 1)]
        ax.plot(list(range(0, 150)), d_s[:, 0].squeeze())
    plt.show()


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


def imshow_mask():
    # _from = nib.load('TPM.nii')
    # print(_from.shape)
    h = nib.load('average_COBRE148subjects.nii')
    # h = nib.load('subject06204_swaurest.nii')
    # h = resample_from_to(_from, _to)
    # h = _from
    print(h.shape)
    # [gray, white, cfs, skull, scalp, other]
    d = h.get_data()
    print(d.min(), d.max())
    frame = 0
    # f = d[:, :, :, frame]
    f = d

    class AnimationFunc:
        s = 0
        im = None

        def __init__(self, img_3d):
            self.img_3d = img_3d

        def __call__(self):
            if self.s < self.img_3d.shape[2]:
                img_2d = self.img_3d[:, :, self.s]
            else:
                print("exit")
                exit()

            print(img_2d.max())
            if self.s == 0:
                self.im = plt.imshow(img_2d, cmap='gray')
                self.im.set_clim(0, 1)
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
    grad_correlation()
