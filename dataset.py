from chainer.dataset import DatasetMixin

import nibabel as nib
from os import listdir
from os import path
import re

class TimeSeriesAutoEncoderDataset(DatasetMixin):
    regexp = re.compile(
        'niftiDATA_Subject(?P<sid>\d{3})_Condition(?P<cid>\d{3}).nii')

    def __init__(self, root):
        self.root = root
        self.files = listdir(self.root)
        self.subjects = [self.regexp.match(file).group('sid') for file in
                         self.files]
        self.subjectnumber = len(self.subjects)
        self.frames = 150

    def __len__(self):
        return self.subjectnumber * self.frames

    def get_example(self, i):
        frame, subject = divmod(i, self.subjectnumber)
        frame = int(frame)
        filepath = path.join(
            self.root, "niftiDATA_Subject{}_Condition000.nii"
                .format(self.subjects[subject]))
        img = nib.load(filepath)
        npimg = img.dataobj[:, :, :, frame]
        return npimg, npimg
