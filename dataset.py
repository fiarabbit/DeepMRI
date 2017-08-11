from chainer.dataset import DatasetMixin
from chainer.datasets import split_dataset
from chainer import Variable

import nibabel as nib
from os import listdir
from os import path
import re


class TimeSeriesAutoEncoderDataset(DatasetMixin):
    regexp = re.compile(
        'niftiDATA_Subject(?P<sid>\d{3})_Condition(?P<cid>\d{3}).nii')
    frame_number = 150

    def __init__(self, root, split_inter=True, subsampling=True,
                 split_ratio=(4, 1)):
        self.root = root
        self.files = listdir(self.root)
        self.subjects = [self.regexp.match(file).group('sid') for file in
                         self.files]
        self.subject_number = len(self.subjects)
        self.split_inter = split_inter
        self.subsampling = subsampling
        self.split_ratio = split_ratio

    def __len__(self):
        return self.subject_number * self.frame_number

    def get_example(self, i):
        if self.split_inter:
            frame, subject = divmod(i, self.subject_number)
        else:
            subject, frame = divmod(i, self.frame_number)

        frame = int(frame)
        filepath = path.join(
            self.root, "niftiDATA_Subject{}_Condition000.nii"
                .format(self.subjects[subject]))
        img = nib.load(filepath)
        npimg = Variable(img.dataobj[:, :, :, frame])
        return npimg, npimg

    def get_subdatasets(self):
        order = []
        if self.split_inter:
            if self.subsampling:
                for i in range(0, len(self)):
                    frame, subject = divmod(i, self.subject_number)
                    if (frame % sum(self.split_ratio)) < self.split_ratio[0]:
                        order.append(i)
            else:
                order = list(
                    range(self.subject_number * self.split_ratio[0] // sum(
                        self.split_ratio) * self.frame_number))
        else:
            order = list(range(self.frame_number * self.split_ratio[0] // sum(
                self.split_ratio) * self.subject_number))
        split_at = len(order)
        assert (split_at != 0) & (split_at != len(self))

        order.extend(set(range(len(self))) - set(order))

        return split_dataset(self, split_at, order)
