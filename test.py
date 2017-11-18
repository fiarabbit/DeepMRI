import chainer as C
from chainer import Variable as V
import chainer.functions as F
import chainer.links as L
import numpy as np
import nibabel as nib

mask = np.array(nib.load("/data/mask/average_optthr.nii").get_data())
mask = mask[mask.nonzero()]
mask = mask.sum()