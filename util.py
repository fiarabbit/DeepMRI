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

from matplotlib.colors import Normalize


def data_correlation():
    root_dir_d = '/data/timeseries'
    root_dir = '/data/mask'

    subjects = ['099', '123', '093', '047', '026', '147', '039', '022', '072',
                '064', '053', '009', '129', '031', '018', '065', '001', '003',
                '071', '024', '030', '102', '017', '139', '074', '020', '070',
                '103', '092', '087', '073', '110', '084', '034', '136', '005',
                '119', '105', '098', '146', '029', '051', '004', '121', '008',
                '138', '112', '133', '097', '044', '055', '088', '125', '108',
                '007', '085', '045', '130', '028', '081', '014', '080', '131',
                '104', '140', '079', '013', '132', '048', '040', '082', '061',
                '058', '086', '068', '095', '075', '010', '122', '056', '046',
                '090', '113', '049', '067', '012', '137', '033', '038', '141',
                '054', '052', '041', '124', '101', '142', '118', '062', '021',
                '066', '109', '037', '019', '126', '096', '115', '100', '120',
                '083', '002', '032', '106', '145', '023', '107', '059', '063',
                '025', '011', '134', '027', '077', '089', '060', '035', '127',
                '057', '094', '128', '050', '036', '091', '006', '143', '116',
                '042', '144', '111', '117', '148', '016', '069', '078', '015',
                '114', '043', '135']

    mask_path = join(root_dir, 'average_optthr.nii')
    base_path = join(root_dir, 'average_COBRE148subjects.nii')

    mask = nib.load(mask_path).get_data()
    base = nib.load(base_path).get_data()

    assert isinstance(mask, np.ndarray)
    assert isinstance(base, np.ndarray)
    assert mask.shape == (91, 109, 91)
    assert base.shape == (91, 109, 91)

    tmp_corr = []
    for subject in subjects:
        file_path = join(root_dir_d,
                         'niftiDATA_Subject{}_Condition000.nii'.format(
                             subject))
        d = nib.load(file_path).get_data()
        assert isinstance(d, np.ndarray)
        try:
            assert d.shape == (91, 109, 91, 150)
        except AssertionError:
            print(d.shape)
            exit()
        d_valid = d[mask.nonzero()]
        assert d_valid.shape == (150350, 150)

        r = np.zeros([150, 150])
        for sample_1 in range(150):
            bold_1 = d_valid[:, sample_1]
            for sample_2 in range(sample_1):
                bold_2 = d_valid[:, sample_2]
                c = np.cov(np.vstack([bold_1, bold_2]))
                r[sample_1, sample_2] = c[0, 1] / np.sqrt(
                    c[0, 0] * c[1, 1])
        tmp_corr.append(r.sum() / np.count_nonzero(r))
    with open("tmpcorr_bold.npz", "wb") as _f:
        np.savez_compressed(_f, data=tmp_corr)
    print(np.mean(tmp_corr))

    # plot
    exit()


def imshow_change_data():
    root_dir = 'C:/Users/hashimoto/PycharmProjects/chainer2python3'
    mask_path = join(root_dir, 'average_optthr.nii')
    base_path = join(root_dir, 'average_COBRE148subjects.nii')

    mask = nib.load(mask_path).get_data()
    base = nib.load(base_path).get_data()

    subject = 0

    with open(join(root_dir, "reconstruction_subject{}.npz".format(subject)),
              "rb") as f:
        d = np.load(f)
        y = d["y"]
        diff = d["diff"]

    with open(join(root_dir, "grad_subject{}.npz".format(subject)), "rb") as f:
        d = np.load(f)
        grad = d["data"]

    print(grad.shape)
    exit()

    # x = np.arange(91 * 109 * 91 * 150).reshape([91, 109, 91, 150])
    # y = x + np.random.rand(x.shape)

    x = y + diff
    print(y.shape)
    print(diff.shape)
    z_list = [33, 38, 43, 48, 53, 58, 63, 68]

    def p(__x):
        print("min: {}\nmax:{}".format(np.min(__x), np.max(__x)))

    def my_cmap(x, cmap, cmin, cmax, threshold, opacity_absolute=False):
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2
        y = cmap(Normalize(cmin, cmax, clip=True)(x))
        alpha = np.copy(x)
        if opacity_absolute:
            cmean = (cmin + cmax) / 2
            alpha = (alpha - cmean) / (cmax - cmean)
            alpha = np.abs(alpha)
        else:
            alpha = (alpha - cmin) / (cmax - cmin)

        alpha = alpha / threshold
        assert isinstance(alpha, np.ndarray)
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        y[..., -1] = alpha
        return y

    imsize = np.array((91, 109))
    figsize = imsize * np.array((len(z_list), 4))
    figsize = np.flip(figsize / np.sqrt(np.sum(figsize**2)) * 15, axis=0)
    figsize = tuple(figsize)
    print(figsize)

    for frame in range(0, 150, 10):
        fig, axeses = plt.subplots(len(z_list), 4)
        fig.set_size_inches(*figsize)
        fig.subplots_adjust(left=0.01, right=1-0.01, bottom=0.01, top=1-0.01, wspace=0.01, hspace=0.01)
        for j, z in enumerate(z_list):
            black = np.zeros(shape=base[:, :, z].shape)
            base_2d = base[:, :, z]
            p(base_2d)
            x_2d = x[frame, :, :, z]
            x_2d_c = my_cmap(x_2d, cmap=plt.cm.RdBu, cmin=-1, cmax=1, threshold=0.5, opacity_absolute=True)
            p(x_2d)
            y_2d = y[frame, :, :, z]
            y_2d_c = my_cmap(y_2d, cmap=plt.cm.RdBu, cmin=-1, cmax=1, threshold=0.5, opacity_absolute=True)
            p(y_2d)
            diff_2d = x_2d - y_2d
            diff_2d_c = my_cmap(diff_2d, cmap=plt.cm.RdBu, cmin=-1, cmax=1, threshold=0.5, opacity_absolute=True)
            p(diff_2d)
            grad_2d = grad[frame, :, :, z]
            grad_2d_c = my_cmap(grad_2d, cmap=plt.cm.hot, cmin=0, cmax=0.0001, threshold=0.5, opacity_absolute=False)
            p(grad_2d)
            # x_masked = x_2d * mask
            # y_masked = y_2d * mask
            axes = axeses[j]
            for i, ax in enumerate(axes):
                ax.set_axis_off()
                if i == 0:
                    ax.imshow(black, cmap='gray')
                    ax.imshow(x_2d_c)
                elif i == 1:
                    ax.imshow(black, cmap='gray')
                    ax.imshow(y_2d_c)
                elif i == 2:
                    ax.imshow(black, cmap='gray')
                    ax.imshow(diff_2d_c)
                else:
                    ax.imshow(base_2d, cmap='gray')
                    ax.imshow(grad_2d_c)
        fig.savefig(join("C:/Users/hashimoto/Pictures/python_output/",
                         "from_left_original_reconstructed_diff_grad_data_subject{}_frame{}.eps".format(
                             subject, frame)),
                    format="eps")


def grad_correlation_inter_subject():
    root_dir_d = '/efs/replication_1000channel/DeepMRI/grad/'
    root_dir = '/data/mask'

    mask_path = join(root_dir, 'average_optthr.nii')
    base_path = join(root_dir, 'average_COBRE148subjects.nii')

    mask = nib.load(mask_path).get_data()
    base = nib.load(base_path).get_data()

    assert isinstance(mask, np.ndarray)
    assert isinstance(base, np.ndarray)
    assert mask.shape == (91, 109, 91)
    assert base.shape == (91, 109, 91)
    d_valid_mean_stack = []
    for i_subject in range(0, 29, 1):
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
            d_valid_mean = np.mean(d_valid, axis=1)
            d_valid_mean_stack.append(d_valid_mean)
    d_valid_mean_stack = np.stack(d_valid_mean_stack)
    assert d_valid_mean_stack.shape == (29, 150350)
    r = np.corrcoef(d_valid_mean_stack)
    print(r)


def grad_correlation():
    root_dir_d = '/efs/replication_1000channel/DeepMRI/grad/'
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
    for i_subject in range(0, 29, 1):
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
                    r[sample_1, sample_2] = c[0, 1] / np.sqrt(
                        c[0, 0] * c[1, 1])
            tmp_corr.append(r.sum() / np.count_nonzero(r))
    with open("tmpcorr.npz", "wb") as _f:
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
        ax.plot(list(range(0, 150)), d_s[:, 1].squeeze())
        ax.set_ylim([-0.13, 0.13])
    plt.show()
    # s = 0
    # d_s = feature[150 * s:150 * (s + 1)]
    # ax.plot(list(range(0, 150)), d_s[:, 0].squeeze())
    # ax.plot(list(range(0, 150)), d_s[:, 1].squeeze())
    # ax.plot(list(range(0, 150)), d_s[:, 2].squeeze())
    # ax.set_ylim([-0.13, 0.13])
    # plt.show()


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
    imshow_change_data()
