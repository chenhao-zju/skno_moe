from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

# from .SKNOBlock import PatchEmbed, SKNOBlock
from SKNOBlock_torch import PatchEmbed, SKNOBlock


class SKNO(nn.Module):
    def __init__(
            self,
            params,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = (params['h_size'], params['w_size'])
        self.patch_size = (params['patch_size'], params['patch_size'])
        self.in_chans = params['feature_dims']
        self.out_chans = params['feature_dims']
        self.num_features = self.embed_dim = params['embed_dim']
        self.num_blocks = params['num_blocks'] 
        self.depth = params['encoder_depths']
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            SKNOBlock(h_size=self.h, w_size=self.w, dim=self.embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(self.depth)])

        # self.norm = norm_layer(self.embed_dim)

        self.head = nn.Linear(self.embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def encoder(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)

        return x
    
    def decoder(self, x):
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

    def forward(self, x):
        x = self.encoder(x)
        print('torch after encoder:', x[0,0,0,:10])

        recons = self.decoder(x)

        for blk in self.blocks:
            x = blk(x)
            print('torch block output:', x.shape, x[0,0,0,:10])
        
        output = self.decoder(x)

        return output, recons

if __name__ == "__main__":
    params = {
        'h_size': 128,
        'w_size': 256,
        'patch_size': 8,
        'feature_dims': 20,
        'num_blocks': 16,
        'encoder_depths': 1,
        'embed_dim': 768
    }
    model = SKNO(params)

    checkpoint = torch.load('/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/checkpoint_ms2torch.ckpt')
    model.load_state_dict(checkpoint)

    # torch.save(model.state_dict(), '/home/bingxing2/ailab/group/ai4earth/haochen/code/skno_moe/test/checkpoint_torch.pth')

    sample = torch.randn(1, 20, 128, 256)
    result = model(sample)
    print(result[0].shape, result[1].shape)
