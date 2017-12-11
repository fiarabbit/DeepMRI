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

import matplotlib.animation as ani

# def grad_anim():
root_dir = 'C:/Users/hashimoto/PycharmProjects/chainer2python3'
file_path = join(root_dir, 'grad.npz')
mask_path = join(root_dir, 'average_optthr.nii')
mask = nib.load(mask_path).get_data()

with open(file_path, "rb") as f:
    data = np.load(f)["grad"]

m = mask * np.abs(data).mean(axis=0)

for i in range(0, m.shape[2], 1):
    s = m[:, :, i]
    if i == 0:
        fig, axes = plt.subplots(1, 2)
    axes[0].imshow(mask[:, :, i])
    im = axes[0].get_images()
    im[0].set_clim(0, 1)
    axes[1].imshow(s)
    im = axes[0].get_images()
    im[0].set_clim(0, 0.0001)
    plt.pause(0.05)


def check_naive():
    datasetdir = '/data/timeseries'
    split_inter = True
    all_dataset = _dataset.TimeSeriesAutoEncoderDataset(datasetdir,
                                                        split_inter=split_inter)
    train_dataset, test_dataset = all_dataset.get_subdatasets()

    mask = np.array(nib.load('/data/mask/average_optthr.nii').get_data())
    mask[mask!=0] = 1

    loss = np.zeros((len(test_dataset),))
    for i, d in enumerate(test_dataset):
        d = d * mask
        loss[i] = np.abs(d.ravel()).mean() * d.size / len(mask.ravel().nonzero()[0])

    print(loss.mean()) # 0.283901693217


def calcoutpsize(insize, kernel, stride, padding):
    return (input+2*padding-kernel)/stride+1


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
                self.im.set_clim(0,2000)
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
    ax.set_xlim(-2,2)
    class Anime():
        k=0
        def __call__(self, i):
            if self.k == 0:
                self.p = ax.bar(b[0:-1], h_list[0, :])
            elif self.k<h_list.shape[0]:
                for i, n in enumerate(self.p):
                    n.set_height(h_list[self.k, i])
            self.k+=1
    a = Anime()
    anim = ani.FuncAnimation(fig, a)
    plt.show()


def plotMeanHist():
    meanArr = np.load('mean_list.npz')['mean_list']
    print(meanArr.mean())
    plt.hist(meanArr)
    plt.show()

if __name__ == '__main__':
    pass