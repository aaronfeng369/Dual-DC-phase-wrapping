import hdf5storage
import numpy as np
import torch.cuda
from tqdm import tqdm
import os
import sys

# Dual-DC phase unwrapping
'''
% path_file: file path (Dual-DC and raw MRE data) (end with /)
% name_: data name (U_<name>.mat)
% lr_unwrap: 0.005,
% iter_max_unwrap:4000,
% gradientDC_coff_unwrap:1000
'''

from phase_wrapping import phase_wrapping_main


if __name__=='__main__':
    # path_file = sys.argv[1]
    # name__ = sys.argv[2]
    # lr_unwrap = float(sys.argv[3])
    # iter_max_unwrap = int(sys.argv[4])
    # gradientDC_coff_unwrap = float(sys.argv[5])

    path_file = ''
    name__ = ['p0','p0.4']
    lr_unwrap = 0.005
    iter_max_unwrap = 4000
    gradientDC_coff_unwrap = 1000

    phase_wrapping_main(path_file, name__, [lr_unwrap,iter_max_unwrap,gradientDC_coff_unwrap])


