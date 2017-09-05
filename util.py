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

arrays = [nib.load(join(datasetdir, files[l])).dataobj for l in range(len_files)]
print(np.stack(arrays, axis=-1).shape)

# for l in range(len_files):
#     path = join(datasetdir, files[l])
#     img = nib.load(path)
#     arrays = []
#     for f in range(len_frames):
#         img.dataobj
#         mean[f, l] = np.mean(img.dataobj[:, :, :, f])
#         std[f, l] = np.std(img.dataobj[:, :, :, f])
#
# np.savez('mean_std.npz', mean=mean, std=std)
# np.savez('mean_std_frame.npz', mean=mean_s, std=std_s)
