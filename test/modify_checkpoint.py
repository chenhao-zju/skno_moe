import torch
import mindspore as ms

src = '/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/checkpoint_ms.ckpt'
des = '/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/checkpoint_ms2torch.ckpt'

checkpoint = ms.load_checkpoint(src)

model = {}

for key, value in checkpoint.items():
    if 'gamma' in key:
        key = key.replace('gamma', 'weight')
        # key = key.replace('gamma', 'bias')
    if 'beta' in key:
        key = key.replace('beta', 'bias')
        # key = key.replace('beta', 'weight')

    value = torch.tensor(value.asnumpy(), dtype=torch.float32)

    if 'filter.sht_cell.weights' in key:
        value = torch.squeeze(value)
        value = value.permute(2,1,0)
        # continue

    if 'filter.isht_cell.weights' in key:
        value = torch.squeeze(value)
        value = value.permute(2,1,0)
        # continue

    model[key] = value

torch.save(model, des)

    



