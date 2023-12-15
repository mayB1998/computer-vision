# main.py
from sfm_pipeline import sfm_pipeline
import numpy as np
import os

# Input Camera Intrinsic Parameters
K = np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047],
              [0, 2398.118540286656, 628.2649953288065],
              [0, 0, 1]])

# Other parameters
bundle_adjustment = False
path = os.getcwd()
img_dir = os.path.join(path, 'buddha')

sfm_pipeline(path, img_dir, K, bundle_adjustment)
