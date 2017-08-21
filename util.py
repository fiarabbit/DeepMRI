import nibabel as nib
import numpy as np
from os import listdir
from os.path import join

datasetdir = '/data/timeseries'
files = listdir(datasetdir)
len_files = len(files)
len_frames = 150
mean = np.empty((len_frames, len_files))
std = np.empty((len_frames, len_files))
mean_s = np.empty((len_files,))
std_s = np.empty((len_files,))

for l in range(len_files):
    path = join(datasetdir, files[l])
    img = nib.load(path)
    # TODO: mask wo tsukeru
    mean_s[l] = np.mean(img.dataobj)
    std_s[l] = np.std(img.dataobj)
    for f in range(len_frames):
        mean[f, l] = np.mean(img.dataobj[:, :, :, f])
        std[f, l] = np.std(img.dataobj[:, :, :, f])

np.savez('mean_std.npz', mean=mean, std=std)
np.savez('mean_std_frame.npz', mean=mean_s, std=std_s)
