from chainer.dataset import DatasetMixin
from chainer.datasets import split_dataset

from os import listdir
from os import path
import pickle
import re
import math


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


class TimeSeriesAutoEncoderDataset(DatasetMixin):
    regexp = re.compile(
        'niftiDATA_Subject(?P<sid>\d{3})'
        + '_Condition(?P<cid>\d{3})'
        + '_frame(?P<fid>\d{3})'
        + '\.pickle$')
    frame_number = 150

    def __init__(self, root, split_inter=True, subsampling=True,
                 split_ratio=(4, 1)):
        self.root = root
        self.files = listdir(self.root)
        self.subjects = unique(
            [self.regexp.match(file).group('sid') for file in
             self.files])
        self.subject_number = len(self.subjects)
        self.split_inter = split_inter
        self.subsampling = subsampling
        self.split_ratio_tuple = split_ratio

    def __len__(self):
        return self.subject_number * self.frame_number

    def get_example(self, i):
        if not self.split_inter:
            frame, subject = divmod(i, self.subject_number)
        else:
            subject, frame = divmod(i, self.frame_number)

        frame = int(frame)
        filepath = path.join(
            self.root, "niftiDATA_Subject{}_Condition000_frame{}.nii.pickle"
                .format(self.subjects[subject], frame))
        with open(filepath, 'rb') as f:
            npimg = pickle.load(f)
        return npimg

    def get_subdatasets(self):
        train = []
        train_ratio_int = self.split_ratio_tuple[0]
        ratio_sum_int = sum(self.split_ratio_tuple)
        if not self.split_inter:
            for i in range(0, len(self)):
                frame, subject = divmod(i, self.subject_number)
                if self.subsampling:
                    if (frame % ratio_sum_int) < train_ratio_int:
                        train.append(i)
                else:
                    threshold = math.ceil(
                        self.frame_number * train_ratio_int / ratio_sum_int
                    )
                    if frame < threshold:
                        train.append(i)
        else:
            for i in range(0, len(self)):
                subject, frame = divmod(i, self.frame_number)
                threshold = math.ceil(
                    self.subject_number * train_ratio_int / ratio_sum_int
                )
                if subject < threshold:
                    train.append(i)
        split_at = len(train)
        assert (split_at != 0) & (split_at != len(self))
        valid = set(range(len(self))) - set(train)
        train.extend(valid)

        return split_dataset(self, split_at, train)
