import numpy as np

import torch
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from SKNO_torch import SKNO as SKNO_torch
from skno_ms import SKNO as SKNO_ms

src_ms = '/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/checkpoint_ms.ckpt'
src_torch = '/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/checkpoint_ms2torch.ckpt'

checkpoint_ms = load_checkpoint(src_ms)
checkpoint_torch = torch.load(src_torch)

B, C, H, W = 16, 20, 128, 256
inputs = np.random.randn(B, C, H, W)

input_ms = ms.Tensor(inputs, dtype=ms.float32)
input_torch = torch.tensor(inputs, dtype=torch.float32)

params = {
        'h_size': H,
        'w_size': W,
        'patch_size': 8,
        'feature_dims': C,
        'num_blocks': 16,
        'encoder_depths': 6,
        'embed_dim': 768
    }

model_ms = SKNO_ms(image_size=(H, W), in_channels=C, out_channels=C, compute_dtype=ms.float32)
load_param_into_net(model_ms, checkpoint_ms)
result_ms = model_ms(input_ms)
print(result_ms[0][0,10,10,5:15], result_ms[1][0,10,10,5:15])

model_torch = SKNO_torch(params)
model_torch.load_state_dict(checkpoint_torch, strict=False)
result_torch = model_torch(input_torch)
print(result_torch[0][0,10,10,5:15], result_torch[1][0,10,10,5:15])




