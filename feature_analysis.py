import numpy as np
import GPy as gp
import sklearn.decomposition

import matplotlib.pyplot as plt


def init():
    global __IPYTHON__, dims, frames, subjects
    try:
        # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
        __IPYTHON__
    except NameError:
        __IPYTHON__ = False
    frames = 150
    dims = (4350, 20,)
    subjects = int(dims[0] / frames)

if __name__=='__main__':
    init()


def loadData(file_path='./feature.npz', check_dims=True):
    global dims
    _data = np.load(file_path)['data']
    if check_dims:
        assert _data.shape == dims
    return _data


def standardize(_data):
    _shape = _data.shape
    _data = np.ravel(_data)
    _data = ((_data - np.mean(_data)) / np.std(_data)).reshape(_shape)
    return _data


def loadLabel():
    global dims
    return np.vstack(
        [x * np.ones((1, dims[1],), dtype=int) for x in range(0, dims[0])])


def PCA():
    global dims, frames, subjects
    data = standardize(loadData())
    label = loadLabel()
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(data)
    x = pca.transform(data)
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    # for subject in range(0, subjects):
    for subject in range(0, 5):
        s = slice(subject * frames, (subject + 1) * frames, None)
        y_1 = x[s, 0]
        y_2 = x[s, 1]
        ax.scatter(y_1, y_2)
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    ax.set_title("subject 0-5")
    ax = axes[1]
    # for frame in range(0, frames):
    for frame in range(0, 5):
        s = slice(frame, dims[0], frames)
        y_1 = x[s, 0]
        y_2 = x[s, 1]
        ax.scatter(y_1, y_2)
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    ax.set_title("frame 0-5")
    plt.show()


def plotTimeseriesInterSubject():
    global frames, dims, subjects
    data = loadData()

    fig, ax = plt.subplots()

    t = range(0, frames)
    feature_idx = 1
    # for subject in range(0, subjects):
    for subject in range(0, 3):
        frame_slice = slice(subject * frames, (subject + 1) * frames, None)
        y = data[frame_slice, feature_idx]
        ax.plot(t, y, label="subject{:02d}".format(subject))
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("feature")
    plt.show()

def plotTimeSeriesIntraSubject():
    global frames, dims, subjects
    data = loadData()
    fig, ax = plt.subplots()

    t = range(0, frames)
    subject = 0
    for feature_idx in range(0,20):
        frame_slice = slice(subject * frames, (subject + 1) * frames, None)
        y = data[frame_slice, feature_idx]
        ax.plot(t, y, label="feature{:02d}".format(feature_idx))

    # plt.legend()
    plt.xlabel("time")
    plt.ylabel("feature")
    plt.show()
plotTimeSeriesIntraSubject()


def GPLVM():
    global latent_dim, data, kernel, model
    latent_dim = 2
    data = standardize(loadData())
    kernel = gp.kern.RBF(latent_dim, ARD=True) \
             + gp.kern.Bias(latent_dim) + gp.kern.White(latent_dim)
    print(kernel)
    model = gp.models.GPLVM(data, latent_dim, kernel=kernel)
    model.optimize(messages=True, max_iters=5e4, ipython_notebook=__IPYTHON__)

def GPLVM_next():
    global frames, dims, subjects
    x = np.load('feature_GPLVM.npz')['feature']
    assert x.shape == (4350, 2)

    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    # for subject in range(0, subjects):
    for subject in range(0, 5):
        s = slice(subject * frames, (subject + 1) * frames, None)
        y_1 = x[s, 0]
        y_2 = x[s, 1]
        ax.scatter(y_1, y_2)
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    ax.set_title("subject 0-5")
    ax = axes[1]
    # for frame in range(0, frames):
    for frame in range(0, 5):
        s = slice(frame, dims[0], frames)
        y_1 = x[s, 0]
        y_2 = x[s, 1]
        ax.scatter(y_1, y_2)
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    ax.set_title("frame 0-5")
    plt.show()

