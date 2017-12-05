from chainer.dataset import DatasetMixin
from chainer.datasets import split_dataset

import nibabel as nib
import numpy as np
from os import listdir
from os import path
import re
import math


class TimeSeriesAutoEncoderDataset(DatasetMixin):
    regexp = re.compile(
        'niftiDATA_Subject(?P<sid>\d{3})_Condition(?P<cid>\d{3})\.nii$')
    frame_number = 150

    def __init__(self, root, mask, split_inter=True, subsampling=True,
                 split_ratio=(4, 1)):
        self.mask = mask
        self.idx_mask = self.mask.nonzero()
        self.root = root
        self.files = listdir(self.root)
        self.subjects = [self.regexp.match(file).group('sid') for file in
                         self.files]
        self.subject_number = len(self.subjects)
        self.split_inter = split_inter
        self.subsampling = subsampling
        self.split_ratio_tuple = split_ratio

    def __len__(self):
        return self.subject_number * self.frame_number

    def get_example(self, i):
        subject, frame = divmod(i, self.frame_number)

        frame = int(frame)
        filepath = path.join(
            self.root, "niftiDATA_Subject{}_Condition000.nii"
                .format(self.subjects[subject]))
        img = nib.load(filepath)
        npimg = img.dataobj[list(self.idx_mask) + [frame]]
        return npimg

    def get_subdatasets(self):
        train = []
        train_ratio_int = self.split_ratio_tuple[0]
        ratio_sum_int = sum(self.split_ratio_tuple)

        for i in range(0, len(self)):
            subject, frame = divmod(i, self.frame_number)
            if self.split_inter:
                threshold = math.ceil(
                    self.subject_number * train_ratio_int / ratio_sum_int
                )
                if subject < threshold:
                    train.append(i)
            else:
                if self.subsampling:
                    if (frame % ratio_sum_int) < train_ratio_int:
                        train.append(i)
                else:
                    threshold = math.ceil(
                        self.frame_number * train_ratio_int / ratio_sum_int
                    )
                    if frame < threshold:
                        train.append(i)

        split_at = len(train)
        assert (split_at != 0) & (split_at != len(self))
        test = set(range(len(self))) - set(train)
        train.extend(test)

        return split_dataset(self, split_at, train)
