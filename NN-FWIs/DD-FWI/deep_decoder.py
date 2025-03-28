# @author: wuyuping (ypwu@stu.hit.edu.cn)

import torch
from torch import nn


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
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64, 64]

        
        self.layer1 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels, hidden_dims[0], 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden_dims[0]),
            )

        self.layer2 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(hidden_dims[0], hidden_dims[1], 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden_dims[1]),
            )
        
        self.layer3 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(hidden_dims[1], hidden_dims[2], 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden_dims[2]),
            )

        self.layer4 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(hidden_dims[2], hidden_dims[3], 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden_dims[3]),
            )

        self.layer5 = nn.Sequential(

                nn.UpsamplingBilinear2d(scale_factor=(1.94,1.625)),
                nn.Conv2d(hidden_dims[3], hidden_dims[4], 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden_dims[4]),
            )

        self.layer6 = nn.Sequential(

                nn.Conv2d(hidden_dims[4], 1, 1),
                # nn.Sigmoid()
            )

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        
        result = self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(z))))))

#         return result
        
        # return 1100 + result[:, :, 0:340, 0:130].contiguous() * (4700.0 - 1100)
        return result[:, :, 0:340, 0:130].contiguous()


    def forward(self, input, **kwargs):

        # :param input: (Tensor) [B x D]

        return self.decode(input)




    