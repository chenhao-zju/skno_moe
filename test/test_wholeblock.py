import torch
import numpy as np
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from SKNOBlock_torch import SKNOBlock as block_torch
from skno_block_ms import SKNOBlock as block_ms

parameters = {'w1': np.random.randn(2,16,48,48), 'b1': np.random.randn(2,16,48), 'w2': np.random.randn(2,16,48,48), 'b2': np.random.randn(2,16,48), 'w.weight': np.random.randn(768,768,1,1)}

parameter_torch = torch.load('/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/ck_block_torch.ckpt')
parameter_ms = load_checkpoint('/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/ck_block_ms.ckpt')


B, C, H, W = 1, 768, 16, 32
inputs = np.random.randn(B, H, W, C)

input_ms = ms.Tensor(inputs, dtype=ms.float32).reshape(B,H*W,C)
input_torch = torch.tensor(inputs, dtype=torch.float32)

model_ms = block_ms(latent_dims=768, mlp_ratio=4, patch_size=8)
load_param_into_net(model_ms, parameter_ms)
result_ms = model_ms(input_ms)
result_ms = result_ms.reshape(B, H, W, C)
print(result_ms[0,0,0,:10])
# print(result_ms[0][0,0,0,:10], result_ms[1][0,0,0,:10])

model_torch = block_torch(h_size=16, w_size=32, dim=768, mlp_ratio=4)
model_torch.load_state_dict(parameter_torch, strict=False)
result_torch = model_torch(input_torch)
print(result_torch[0,0,0,:10])
# print(result_torch[0][0,0,0,:10], result_torch[1][0,0,0,:10])


