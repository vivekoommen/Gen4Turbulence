Please ensure that hr_data.npy and lr8_data.npy are in this directory.

hr_data  - [nt=1000, nx=128, ny=256]
lr8_data - [nt=1000, nx=128, ny=256]

lr8_data was subsampled in space by a factor of 8 and then interpolated back to original resolution of [128,256] in data_prep.ipynb

The training scripts for the various models in 1_superresolution/no/* directly subsamples time by a factor of 4 

Please contact Prof. He Feng (hefeng@tsinghua.edu.cn) to access this dataset.