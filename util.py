import nibabel as nib
import numpy as np
from os import listdir
from os.path import join

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()
import time

import matplotlib.animation as ani

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


def hoge():
    # x = nib.load('niftiDATA_Subject001_Condition000.nii').dataobj[:,:,45,0]
    # x = np.load('hoge.npz')['x'].reshape((91,109,91))[:,:,45]
    import pickle
    with open('niftiDATA_Subject001_Condition000_frame000.pickle', 'rb') as f:
        img = pickle.load(f)
    x = img[:,:,45]
    y = np.load('hoge.npz')['y2'].reshape((91,109,91))[:,:,45]
    plt.subplot(131)
    ax_x = plt.imshow(x)
    ax_x.set_clim(0,1)
    plt.subplot(132)
    ax_y = plt.imshow(y)
    ax_y.set_clim(0,1)
    plt.subplot(133)
    ax_scale = plt.imshow(x*y)
    ax_scale.set_clim(0,1)
    plt.show()

def mask():
    h = nib.load('niftiDATA_Subject001_Condition000.nii')
    # h = nib.load('average_optthr.nii')
    # h = nib.load('subject06204_swaurest.nii')
    d = h.dataobj
    frame = 0
    f = d[:, :, :, frame]
    # f = np.load('hoge.npz')['x']*np.load('hoge.npz')['y2']
    # f = np.load('hoge.npz')['y2']
    # f = f.reshape((91, 109, 91))
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
                self.im.set_clim(0,1)
            elif self.s > 0:
                self.im.set_data(img_2d)
            plt.pause(0.1)
            self.s += 1
            print(self.s)
            if self.s == self.img_3d.shape[2]:
                raise StopIteration
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
    hoge()