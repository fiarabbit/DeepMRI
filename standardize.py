import argparse
import os

import nibabel as nib
import numpy as np
import numpy.ma as ma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='/data/timeseries')
    parser.add_argument('--mask', default='/data/mask/average_optthr.nii')
    parser.add_argument('--result', default='/data2/preprocessed_data')
    args = parser.parse_args()
    mask = nib.load(args.mask).get_data()
    mask = np.reshape(mask, [1] + mask.shape)
    targets = os.listdir(args.target)
    for i in range(targets):
        target = targets[i]
        print("processing {}/{}".format(i, len(targets)))
        t = os.path.join(args.target, target)
        x = nib.load(t).get_data()
        if mask.shape != x.shape:
            mask = np.broadcast_to(mask, x.shape)
        x_masked = ma.masked_where(mask, x)
        x_voxel \
            = x_masked - np.mean(x_masked, axis=x_masked.shape[-1])
        x_standardized = (x_voxel - np.mean(x_voxel)) / np.std(x_voxel)
        np.savez_compressed(
            os.path.join(args.result, target), dataobj=x_standardized)


if __name__ == '__main__':
    main()
