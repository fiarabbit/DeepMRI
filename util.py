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

for l in range(len_files):
    path = join(datasetdir, files[l])
    img = nib.load(path)
    for f in range(len_frames):
        mean[f, l] = np.mean(img.dataobj[:, :, :, f])
        std[f, l] = np.std(img.dataobj[:, :, :, f])

np.savez('mean_std.npz' ,mean, std)
