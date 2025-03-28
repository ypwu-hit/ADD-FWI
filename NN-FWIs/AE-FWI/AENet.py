# @author: wuyuping (ypwu@stu.hit.edu.cn)

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import numpy as  np


class CNN_Encoder(nn.Module):
    def __init__(self, channel_init, embedding_size, in_channels, input_height, input_width):
        super(CNN_Encoder, self).__init__()

        self.input_size = (in_channels, input_height, input_width)
        self.channel_mult = channel_init

        #convolutions
        self.conv = nn.Sequential(
            # e1
            nn.Conv2d(in_channels, self.channel_mult * 1, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult*1, self.channel_mult*1, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            # e2
            nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult*2, self.channel_mult*2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            # e3
            nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 4, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True),

            # e4
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult * 8, self.channel_mult * 8, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2, ceil_mode=True)

        )

        # 这个self.flat_fts相当于channels * input_height * input_width
        self.flat_fts, self.conv_channels, self.conv_height, self.conv_width = self.get_flat_fts(self.conv)

        # [batchsize, conv_channels * conv_height * conv_width]
        # 转换成为[batchsize, embedding_size]
        self.linear_embedding = nn.Sequential(
            nn.Linear(self.flat_fts, embedding_size),
            nn.ReLU()
        )

        # nn.BatchNorm1d(self.flat_fts * 2) 这个的使用需要有Batchsize大于1
        # 将[batchsize, embedding_size]转换为[batchsize, conv_channels * conv_height * conv_width * 2]
        self.linear_reconstruct = nn.Sequential(
            nn.Linear(embedding_size, self.flat_fts * 2),
            nn.ReLU()
        )

    def get_flat_fts(self, fts):
        f = fts(Variable(torch.ones(1, *self.input_size)))

        # 相当于计算编码器输出的四维张量中后三维相乘的大小[batchsize, channels, input_height, input_width]
        # return channels * input_height * input_width
        return int(np.prod(f.size()[1:])), int(f.size()[1]), int(f.size()[2]), int(f.size()[3])

    def forward(self, x):
        # 相当于输入是[batchsize, in_channels, input_height, input_width]
        x = self.conv(x)
        # print('x.cpu().shape, self.conv(x)', x.cpu().shape)

        # x.view的输出 [batchsiez, channels * conv_height * conv_width]
        x = x.view(-1, self.flat_fts)
        # print('x.view(-1, self.flat_fts)', x.cpu().shape)

        # [batchsize, embedding_size]
        x = self.linear_embedding(x)
        # print('x.cpu().shape, self.linear_embedding(x)', x.cpu().shape)

        # x = self.linear_reconstruct(x).view(-1, self.conv_channels*2, self.conv_height, self.conv_width)
        # print('self.linear_reconstruct(x)', x.cpu().shape)

        # return [batchsize, conv_channels * conv_height * conv_width * 2]
        return self.linear_reconstruct(x).view(-1, self.conv_channels*2, self.conv_height, self.conv_width)

class CNN_Decoder(nn.Module):
    def __init__(self, channel_init, embedding_size, output_channels, input_height, input_width):
        super(CNN_Decoder, self).__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.input_dim = embedding_size
        self.channel_mult = channel_init
        self.output_channels = output_channels
        self.fc_output_dim = self.channel_mult*16

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            # d4
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.fc_output_dim, self.channel_mult * 8, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult * 8, self.channel_mult * 8, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.ReLU(inplace=True),

            # d3
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channel_mult * 8, self.channel_mult * 4, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 4, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(inplace=True),

            # d2
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 2, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.ReLU(inplace=True),

            # d1
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 1, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.channel_mult * 1, self.channel_mult * 1, 3, 1, 1),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(inplace=True),

            # final
            nn.Conv2d(self.channel_mult * 1, self.output_channels, 1),

        )

    def forward(self, x):

        x = self.deconv(x)
        # print('self.deconv(x)', x.cpu().shape)

        return x[:, :, 0:340, 0:130].contiguous()


class Network(nn.Module):
    def __init__(self, channel_init,embedding_size,in_channels,output_channels,input_height,input_width):
        super(Network, self).__init__()

        self.encoder = CNN_Encoder(channel_init, embedding_size,
                                   in_channels, input_height, input_width)

        self.decoder = CNN_Decoder(channel_init, embedding_size,
                                   output_channels, input_height, input_width)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


