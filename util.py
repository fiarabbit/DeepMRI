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
import json
import re
from os import path


def create_npimg(root_dir='/data/timeseries',
                 mask_path='/data/mask/average_optthr.nii',
                 dest_dir='/data3/data'):

    idx_mask = np.array(nib.load(mask_path).get_data()).nonzero()

    regexp = re.compile(
        'niftiDATA_Subject(?P<sid>\d{3})_Condition(?P<cid>\d{3})\.nii$')
    frame_number = 150
    subjects = [regexp.match(file).group('sid') for file in listdir(root_dir)]
    for subject in subjects:
        print(subject)
        filepath = path.join(root_dir, "niftiDATA_Subject{}_Condition000.nii"
                             .format(subject))
        npimg = np.array(nib.load(filepath).get_data())
        for frame in range(frame_number):
            print(frame)
            print("subject{:03d}_frame{:03d}.npy".format(int(subject), frame))
            dest_filepath = path.join(dest_dir, "subject{0:03d}_frame{0:03d}.npy".format(int(subject), frame))
            npimg_sliced = npimg[list(idx_mask) + [frame]]
            np.save(dest_filepath, npimg_sliced)


"""mask
valid = mask[10:80, 11:99, 3:77]
150350 True / 455840 (= 70 * 88 * 74)
"""

"""plan1 - VGG-like w/ Init=Identity
Principally, the initial weights do just an aliasing.
So the results cannot be worth than "just averaging"

Note: learn with drop-out

1, 70, 88, 74
conv1 = Identity (0, 1, 0) -> (1/64,)
64, 70, 88, 74
conv2 = Identity (0, 1, 0) -> (1,)
64, 70, 88, 74
pool1
64, 35, 44, 37
conv3 = Identity
128, 36, 44, 38
conv4 = Identity
128, 36, 44, 38
pool2
128, 18, 22, 19
conv5 = Identity
256, 18, 22, 20
conv6 = Identity
256, 18, 22, 20
pool3
256, 9, 11, 10
conv7 = Identity
512, 10, 12, 10
conv8 = Identity
512, 10, 12, 10
pool4
512, 5, 6, 5
conv9 = Identity
1024, 6, 6, 6
conv10 = Identity
1024, 6, 6, 6
pool5
1024, 3, 3, 3
1*1conv = Constant (1/1024,) -> (1/k,)
k, 3, 3, 3 (k1=27, k10=270, k100=2700)
1*1deconv = Constant (1/k,) -> (1/1024)
1024, 3, 3, 3
upsample5
1024, 6, 6, 6
conv10 = Identity ()
1024, 6, 6, 6
deconv9 = Identity (1/1024) -> (0, 1/512, 0)
512, 5, 6, 5
upsample4
512, 10, 12, 10
deconv8 = Identity
512, 10, 12, 10
deconv7 = Identity
256, 9, 11, 10
upsample3
256, 18, 22, 20
deconv6 = Identity
256, 18, 22, 20
deconv5 = Identity
128, 18, 22, 19
upsample2
128, 36, 44, 38
deconv4 = Identity
128, 36, 44, 38
deconv3 = Identity
64, 35, 44, 37
upsample1
64, 70, 88, 74
deconv2 = Identity
64, 70, 88, 74
deconv1 = Identity
1, 70, 88, 74
"""

"""plan2 - MLP
1, 70, 88, 74
mask
150350,
linear
1000,
linear
150350,
unmask
1, 70, 88, 74
"""

"""plan3 - VGG-like w/ Init=Identity
Principally, the initial weights do just an aliasing.
So the results cannot be worth than "just averaging"

Note: learn with drop-out

1, 70, 88, 74
conv1 = Identity (0, 1, 0) -> (1/64,)
64, 70, 88, 74
conv2 = Identity (0, 1, 0) -> (1,)
64, 70, 88, 74
pool1
64, 35, 44, 37
conv3 = Identity
128, 36, 44, 38
conv4 = Identity
128, 36, 44, 38
pool2
128, 18, 22, 19
conv5 = Identity
256, 18, 22, 20
conv6 = Identity
256, 18, 22, 20
pool3
256, 9, 11, 10
conv7 = Identity
512, 10, 12, 10
conv8 = Identity
512, 10, 12, 10
pool4
512, 5, 6, 5
conv9 = Identity
1024, 6, 6, 6
conv10 = Identity
1024, 6, 6, 6
pool5
1024, 3, 3, 3
1*1conv = Constant (1/1024,) -> (1/k,)
k, 3, 3, 3 (k1=27, k10=270, k100=2700)
1*1deconv = Constant (1/k,) -> (1/1024)
1024, 3, 3, 3
upsample5
1024, 6, 6, 6
conv10 = Identity ()
1024, 6, 6, 6
deconv9 = Identity (1/1024) -> (0, 1/512, 0)
512, 5, 6, 5
upsample4
512, 10, 12, 10
deconv8 = Identity
512, 10, 12, 10
deconv7 = Identity
256, 9, 11, 10
upsample3
256, 18, 22, 20
deconv6 = Identity
256, 18, 22, 20
deconv5 = Identity
128, 18, 22, 19
upsample2
128, 36, 44, 38
deconv4 = Identity
128, 36, 44, 38
deconv3 = Identity
64, 35, 44, 37
upsample1
64, 70, 88, 74
deconv2 = Identity
64, 70, 88, 74
deconv1 = Identity
1, 70, 88, 74
"""


def show_log():
    with open('log', 'r') as f:
        data = json.load(f)

    val_loss = np.zeros((len(data),))
    for i, item in enumerate(data):
        val_loss[i] = item['validation/main/loss']
    fig, ax = plt.subplots(1, 1)
    plt.plot(val_loss[5:])
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
    create_npimg()
