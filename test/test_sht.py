import numpy as np
import mindspore as ms
import torch

from SKNOBlock_torch import RealSHT as sht_torch
from SKNOBlock_torch import InverseRealSHT as isht_torch
from skno_block_ms import RealSHT as sht_ms
from skno_block_ms import InverseRealSHT as isht_ms

inputs = np.random.randn(1,20,32,64)

inputs_ms = ms.Tensor(inputs, dtype=ms.float32)
sht_ms = sht_ms(nlat=32, nlon=64, lmax=32, mmax=33)
mid_ms = sht_ms(inputs_ms)

isht_ms = isht_ms(nlat=32, nlon=64, lmax=32, mmax=33)
output_ms = isht_ms(mid_ms)


inputs_torch = torch.tensor(inputs, dtype=torch.float32)
sht_torch = sht_torch(nlat=32, nlon=64, lmax=32, mmax=33)
mid_torch = sht_torch(inputs_torch)

isht_torch = isht_torch(nlat=32, nlon=64, lmax=32, mmax=33)
output_torch = isht_torch(mid_torch)

print('input feature: ', inputs.shape, inputs[0,:10,0,0])
print('output feature: ', output_ms.shape, output_ms[0,:10,0,0], output_torch[0,:10,0,0])



