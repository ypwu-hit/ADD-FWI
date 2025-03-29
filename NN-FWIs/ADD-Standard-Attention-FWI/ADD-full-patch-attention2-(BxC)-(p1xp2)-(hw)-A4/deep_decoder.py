# @author: wuyuping (ypwu@stu.hit.edu.cn)

import torch
from torch import nn
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F

# Copied from https://github.com/wilson1yan/VideoGPT
def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0.0, training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x d

    return a

class MultiHeadAttention_patch2(nn.Module):
    def __init__(
        self, dim=11*5, n_head=8, h=11, w=5, c=64, p1=2, p2=2
    ):
        super().__init__()

        self.d = dim // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim, n_head * self.d, bias=False)  # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim))

        self.w_ks = nn.Linear(dim, n_head * self.d, bias=False)  # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim))

        self.w_vs = nn.Linear(dim, n_head * self.d, bias=False)  # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim))

        self.fc = nn.Linear(n_head * self.d, dim, bias=True)  # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim))

        self.rerange1 = nn.Sequential(
            Rearrange("b c (p1 h) (p2 w) -> (b c) (p1 p2) (h w)", p1=p1, p2=p2, h=h, w=w),
        )

        self.rerange2 = nn.Sequential(
            Rearrange("(b c) (p1 p2) (h w) -> b c (p1 h) (p2 w)", p1=p1, p2=p2, c=c, h=h, w=w),        
        )
    
    def forward(self, x):
        
        # self.rerange(x) -> (b c h w) -> ((b c) (p1 p2) (h w))
        q = self.w_qs(self.rerange1(x))
        k = self.w_ks(self.rerange1(x))
        v = self.w_vs(self.rerange1(x))
        
        out = scaled_dot_product_attention(q, k, v)
        out = self.fc(out)
        
        return self.rerange2(out)


class VanillaConvUpDecoder(nn.Module):


    def __init__(self,
                 in_channels=64,
                 latent_out_dim_h=11,
                 latent_out_dim_w=5,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(VanillaConvUpDecoder, self).__init__()

        self.in_channels = in_channels
        self.latent_out_dim_h = latent_out_dim_h
        self.latent_out_dim_w = latent_out_dim_w
        self.n = latent_out_dim_h * latent_out_dim_w
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64, 64]

        # step 1
        # B C H W
        self.up1 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2)
                )

        self.patch_attention_1 = nn.Sequential(
                MultiHeadAttention_patch2(dim=11*5, n_head=8, h=11,w=5,c=64,p1=2,p2=2),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )

        # B C H W -> (B C) (H/P1 W/P2) p1 p2
        # self.patch_attention_1 = nn.Sequential(
        #         Rearrange("b c (p1 h) (p2 w) -> (b c) (p1 p2) h w", p1=2, p2=2, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.Conv2d(4, 4, 1),
        #         Rearrange("(b c) (p1 p2) h w -> b c (p1 h) (p2 w)", p1=2, p2=2, b=1, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.BatchNorm2d(in_channels),
        #         nn.ReLU(inplace=True)
        #         )

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], 1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )

        # step 2
        # B C H W
        self.up2 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2)
                )

        self.patch_attention_2 = nn.Sequential(
                MultiHeadAttention_patch2(dim=11*5, n_head=8, h=11,w=5,c=64,p1=4,p2=4),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
                )

        # B C 44 20
        # B C H W -> (B C) (H/P1 W/P2) p1 p2
        # self.patch_attention_2 = nn.Sequential(
        #         Rearrange("b c (p1 h) (p2 w) -> (b c) (p1 p2) h w", p1=4, p2=4, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.Conv2d(16, 16, 1),
        #         Rearrange("(b c) (p1 p2) h w -> b c (p1 h) (p2 w)", p1=4, p2=4, b=1, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.BatchNorm2d(hidden_dims[0]),
        #         nn.ReLU(inplace=True)
        #         )

        self.layer2 = nn.Sequential(
                nn.Conv2d(hidden_dims[0], hidden_dims[1], 1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.ReLU(inplace=True)
            )

        # step 3
        # B C H W
        self.up3 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2)
                )

        # B C H W -> (B C) (H/P1 W/P2) p1 p2
        # self.patch_attention_3 = nn.Sequential(
        #         Rearrange("b c (p1 h) (p2 w) -> (b c) (h w) p1 p2", p1=8, p2=8, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.Conv2d(self.n, self.n, 1),
        #         Rearrange("(b c) (h w) p1 p2 -> b c (p1 h) (p2 w)", b=1, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.BatchNorm2d(hidden_dims[1]),
        #         nn.ReLU(inplace=True)
        #         )

        self.layer3 = nn.Sequential(
                nn.Conv2d(hidden_dims[1], hidden_dims[2], 1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.ReLU(inplace=True)
            )

        # step 4
        # B C H W
        self.up4 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2)
                )

        # B C H W -> (B C) (H/P1 W/P2) p1 p2
        # self.patch_attention_4 = nn.Sequential(
        #         Rearrange("b c (p1 h) (p2 w) -> (b c) (h w) p1 p2", p1=16, p2=16, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.Conv2d(self.n, self.n, 1),
        #         Rearrange("(b c) (h w) p1 p2 -> b c (p1 h) (p2 w)", b=1, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.BatchNorm2d(hidden_dims[2]),
        #         nn.ReLU(inplace=True)
        #         )

        self.layer4 = nn.Sequential(
                nn.Conv2d(hidden_dims[2], hidden_dims[3], 1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.ReLU(inplace=True)
            )

        # step 5
        # B C H W
        self.up5 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=(1.94, 1.625))
                )

        # B C H W -> (B C) (H/P1 W/P2) p1 p2
        # self.patch_attention_5 = nn.Sequential(
        #         Rearrange("b c (p1 h) (p2 w) -> (b c) (h w) p1 p2", p1=16*1.94, p2=16*1.625, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.Conv2d(self.n, self.n, 1),
        #         Rearrange("(b c) (h w) p1 p2 -> b c (p1 h) (p2 w)", b=1, h=latent_out_dim_h, w=latent_out_dim_w),
        #         nn.BatchNorm2d(hidden_dims[3]),
        #         nn.ReLU(inplace=True)
        #         )

        self.layer5 = nn.Sequential(
                nn.Conv2d(hidden_dims[3], hidden_dims[4], 1),
                nn.BatchNorm2d(hidden_dims[4]),
                nn.ReLU(inplace=True)
            )
        

        self.layer6 = nn.Sequential(

                nn.Conv2d(hidden_dims[4], 1, 1),
            )

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        outputs = self.layer1(self.patch_attention_1(self.up1(z)))
        outputs = self.layer2(self.patch_attention_2(self.up2(outputs)))
        outputs = self.layer3(self.up3(outputs))
        outputs = self.layer4(self.up4(outputs))
        outputs = self.layer5(self.up5(outputs))
        outputs = self.layer6(outputs)
        
        return outputs[:, :, 0:340, 0:130].contiguous()


    def forward(self, input, **kwargs):

        # :param input: (Tensor) [B x D]

        return self.decode(input)




    