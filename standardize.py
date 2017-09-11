import argparse
import os

import nibabel as nib
import numpy as np
import numpy.ma as ma

import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='/data/timeseries')
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    parser.add_argument('--result', default='/data2/preprocessed_data')
    args = parser.parse_args()
    mask = nib.load(args.mask).get_data()
    mask = np.reshape(mask, list(mask.shape) + [1])
    targets = os.listdir(args.target)
    for i in range(len(targets)):
        target = targets[i]
        print("processing {}/{}".format(i, len(targets)))
        t = os.path.join(args.target, target)
        x = nib.load(t).get_data()
        if mask.shape != x.shape:
            mask = np.broadcast_to(mask, x.shape)
        x_masked = ma.masked_where(mask, x)
        _x_mean = np.mean(x_masked, axis=-1)
        x_mean = np.reshape(_x_mean, list(_x_mean.shape)+[1])
        x_voxel \
            = x_masked - x_mean
        x_standardized = (x_voxel - np.mean(x_voxel)) / np.std(x_voxel)
        assert isinstance(x_standardized, ma.MaskedArray)
        save_name = re.sub("$", ".npz", target)
        save_path = os.path.join(args.result, save_name)
        print(save_path)
        np.savez_compressed(save_path, data=ma.filled(x_standardized, 0))
        # x_standardized.dump(os.path.join(args.result, save_name))


if __name__ == '__main__':
    main()
