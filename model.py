#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Construct the computational graph of model for training. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding,stride):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU()))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU()))
        elif char == 'e':
            modules.append(('ELU', nn.ELU()))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size ,stride=stride, bias = bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cgr', num_groups=8, padding=(1,1,1),stride = (1,1,1)):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding, stride=stride):
            self.add_module(name, module)

class Devideconv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(Devideconv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, (kernel_size,1,1), order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, (1,kernel_size,kernel_size), order, num_groups,
                                   padding=padding))

class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))

class Inception_st(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels_1, kernel_s=3,kernel_t=9, order='gcr', num_groups=8, **kwargs):
        super(Inception_st, self).__init__()
        out_channels = int(out_channels_1 / 4)
        middle_channels = int(out_channels_1 / 2)
        # first convolution
        self.conv11 = SingleConv(in_channels, middle_channels, kernel_size=(1,1,1), order=order, num_groups=num_groups,
                                 padding=(0, 0, 0))
        self.conv12 = SingleConv(middle_channels, middle_channels, kernel_size=(1,kernel_s,kernel_s), order=order, num_groups=num_groups,
                                 padding=(0, int((kernel_s-1)/2), int((kernel_s-1)/2)))
        self.conv13 = SingleConv(middle_channels, out_channels, kernel_size=(kernel_t,1,1), order=order, num_groups=num_groups,
                                 padding=(int((kernel_t-1)/2), 0, 0))

        self.conv21 = SingleConv(in_channels, middle_channels, kernel_size=(1,1,1), order=order, num_groups=num_groups,
                                 padding=(0, 0, 0))
        self.conv22 = SingleConv(middle_channels, middle_channels, kernel_size=(kernel_t,1,1), order=order, num_groups=num_groups,
                                 padding=(int((kernel_t-1)/2), 0, 0))
        self.conv23 = SingleConv(middle_channels, out_channels, kernel_size=(1,kernel_s,kernel_s), order=order, num_groups=num_groups,
                                 padding=(0, int((kernel_s-1)/2), int((kernel_s-1)/2)))

        self.conv31 = SingleConv(in_channels, middle_channels, kernel_size=(1,1,1), order=order, num_groups=num_groups,
                                  padding=(0, 0, 0))
        self.conv32 = SingleConv(middle_channels, middle_channels, kernel_size=(1,kernel_s,kernel_s), order=order, num_groups=num_groups,
                                 padding=(0, int((kernel_s-1)/2), int((kernel_s-1)/2)))
        self.conv33 = SingleConv(middle_channels, middle_channels, kernel_size=(kernel_t,1,1), order=order, num_groups=num_groups,
                                 padding=(int((kernel_t-1)/2), 0, 0))
        self.conv34 = SingleConv(middle_channels, out_channels, kernel_size=(1,kernel_s,kernel_s), order=order, num_groups=num_groups,
                                 padding=(0, int((kernel_s-1)/2), int((kernel_s-1)/2)))

        self.conv41 = SingleConv(in_channels, out_channels, kernel_size=(1,1,1), order=order, num_groups=num_groups,
                                 padding=(0, 0, 0))

        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1)
        elif 'e' in order:
            self.non_linearity = nn.ELU()
        else:
            self.non_linearity = nn.ReLU()

    def forward(self, x):
        # apply first convolution and save the output as a residual
        x1 = self.conv13(self.conv12(self.conv11(x)))
        x2 = self.conv23(self.conv22(self.conv21(x)))
        x3 = self.conv34(self.conv33(self.conv32(self.conv31(x))))
        x4 = self.conv41(x)

        out = torch.cat((x1,x2,x3,x4),1)
        del x1,x2,x3,x4
        torch.cuda.empty_cache()

        return out

class ResBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8,padding=1, **kwargs):
        super(ResBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups, padding=padding)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups, padding=padding)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)
        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1)
        elif 'e' in order:
            self.non_linearity = nn.ELU()
        else:
            self.non_linearity = nn.ReLU()

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out = out + residual
        out = self.non_linearity(out)

        return out

class InterpolateUpsampling(nn.Module):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode = 'nearest'):
        super(InterpolateUpsampling, self).__init__()
        self.mode = mode


    def forward(self, x):
        # get the spatial dimensions of the output given the encoder_features
        x_size = x.size()
        output_size = []
        for i in range(2,len(x_size)):
            output_size.append(int(x_size[i]*2))

        # upsample the input and return
        return F.interpolate(x, size=output_size, mode=self.mode)

def create_transconv(in_channels, out_channels, kernel_size=3, stride=(2,2,2), padding=1, order = 'cgr', num_groups = 4):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> transconv + ReLU
            'gcr' -> groupnorm + transconv + ReLU
            'cl' -> transconv + LeakyReLU
            'ce' -> transconv + ELU
            'bcr' -> batchnorm + transconv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Transconv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU()))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU()))
        elif char == 'e':
            modules.append(('ELU', nn.ELU()))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class TransposeSingleConv(nn.Sequential):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
        out_channels (int): number of output channels for transpose conv
        kernel_size (int or tuple): size of the deconvolving kernel
        stride (int or tuple): stride of the deconvolution

    """
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2), padding=(2,1,1), order = 'cgr', num_groups=8):
        super(TransposeSingleConv, self).__init__()

        for name, module in create_transconv(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                             padding=padding, order = order, num_groups= num_groups):
            self.add_module(name, module)
# class Feature_extractor(nn.Module):
#     def __init__(self, kernel_size_1 = 3, kernel_size_2 = 3):
#         super(Feature_extractor, self).__init__()
#         self.conv_time1 = SingleConv(1, 2, kernel_size=(kernel_size_1,1,1), order='cgr', num_groups=1, padding=(int((kernel_size_1-1)/2) ,0, 0))
#         self.conv_space1 = SingleConv(2, 2, kernel_size=(1, kernel_size_1, kernel_size_1), order='cgr', num_groups=1,
#                                       padding=(0, int((kernel_size_1-1)/2), int((kernel_size_1-1)/2)))
#         self.conv_res1 = ResBlock(2, 4, kernel_size=kernel_size_1, order='cgr', num_groups=2, padding=int((kernel_size_1-1)/2))
#
#         self.conv_time2 = SingleConv(4, 8, kernel_size=(kernel_size_2,1,1), order='cgr', num_groups=2, padding=(int((kernel_size_2-1)/2) ,0, 0))
#         self.conv_space2 = SingleConv(8, 8, kernel_size=(1, kernel_size_2, kernel_size_2), order='cgr', num_groups=2,
#                                       padding=(0, int((kernel_size_2-1)/2), int((kernel_size_2-1)/2)))
#         self.conv_res2 = ResBlock(8, 16, kernel_size=kernel_size_2, order='cgr', num_groups=4, padding=int((kernel_size_2-1)/2))
#
#         self.conv_time3 = SingleConv(16, 32, kernel_size=(kernel_size_2,1,1), order='cgr', num_groups=8, padding=(int((kernel_size_2-1)/2) ,0, 0))
#         self.conv_space3 = SingleConv(32, 32, kernel_size=(1, kernel_size_2, kernel_size_2), order='cgr', num_groups=8,
#                                       padding=(0, int((kernel_size_2-1)/2), int((kernel_size_2-1)/2)))
#         self.conv_res3 = ResBlock(32, 64, kernel_size=kernel_size_2, order='cgr', num_groups=8, padding=int((kernel_size_2-1)/2))
#         self.pooling = nn.MaxPool3d(kernel_size=2)
#
#         self.conv_time4 = SingleConv(64, 64, kernel_size=(kernel_size_2,1,1), order='cgr', num_groups=8, padding=(int((kernel_size_2-1)/2) ,0, 0))
#         self.conv_space4 = SingleConv(64, 64, kernel_size=(1, kernel_size_2, kernel_size_2), order='cgr', num_groups=8,
#                                       padding=(0, int((kernel_size_2-1)/2), int((kernel_size_2-1)/2)))
#         self.conv_res4 = ResBlock(64, 64, kernel_size=kernel_size_2, order='cgr', num_groups=16, padding=int((kernel_size_2-1)/2))
#
#     def forward(self, x):
#         # x = self.conv0(x)
#         t1 = self.conv_time1(x)
#         s1 = self.conv_space1(t1)
#         r1 = self.conv_res1(s1)
#         p1 = self.pooling(r1)
#
#         t2 = self.conv_time2(p1)
#         s2 = self.conv_space2(t2)
#         r2 = self.conv_res2(s2)
#         p2 = self.pooling(r2)
#
#         t3 = self.conv_time3(p2)
#         s3 = self.conv_space3(t3)
#         r3 = self.conv_res3(s3)
#         p3 = self.pooling(r3)
#
#         t4 = self.conv_time4(p3)
#         s4 = self.conv_space4(t4)
#         r4 = self.conv_res4(s4)
#
#         return r1,r2,r3,r4

class Feature_extractor(nn.Module):
    def __init__(self, kernel_s = 3, kernel_t = 9):
        super(Feature_extractor, self).__init__()

        self.conv_st_inc1 = Inception_st(1, 4, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=1)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(2,1,1))
        self.conv_st_inc2 = Inception_st(4, 8, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=1)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(2,1,1))
        self.conv_st_inc3 = Inception_st(8, 12, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=1)
        self.pool3 = torch.nn.MaxPool3d(kernel_size=(2,1,1))
        self.conv_st_inc4 = Inception_st(12, 16, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=2)
        self.pool4 = torch.nn.MaxPool3d(kernel_size=(2,1,1))#240
        self.conv_st_inc5 = Inception_st(16, 24, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=2)
        self.pool5 = torch.nn.MaxPool3d(kernel_size=(2,1,1))
        self.conv_st_inc6 = Inception_st(24, 32, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=2)
        self.pool6 = torch.nn.MaxPool3d(kernel_size=(2,1,1))
        self.conv_st_inc7 = Inception_st(32, 40, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=2)
        self.pool7 = torch.nn.MaxPool3d(kernel_size=(2,1,1))
        self.conv_st_inc8 = Inception_st(40, 48, kernel_s=kernel_s, kernel_t=kernel_t, order='cgr', num_groups=2)

    def forward(self, x):
        x = self.conv_st_inc1(x)
        x = self.pool1(x)
        x = self.conv_st_inc2(x)
        x = self.pool2(x)
        x = self.conv_st_inc3(x)
        x = self.pool3(x)
        x = self.conv_st_inc4(x)
        x = self.pool4(x)
        x = self.conv_st_inc5(x)
        x = self.pool5(x)
        x = self.conv_st_inc6(x)
        x = self.pool6(x)
        x = self.conv_st_inc7(x)
        x = self.pool7(x)
        x = self.conv_st_inc8(x)
        torch.cuda.empty_cache()
        return x

class Reconstructor(nn.Module):
    def __init__(self, kernel_size=(6,3,3)):
        super(Reconstructor, self).__init__()

        self.up_conv1 = TransposeSingleConv(48, 40, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.up_conv2 = TransposeSingleConv(40, 32, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.up_conv3 = TransposeSingleConv(32, 24, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.up_conv4 = TransposeSingleConv(24, 16, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.up_conv5 = TransposeSingleConv(16, 12, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.up_conv6 = TransposeSingleConv(12, 8, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.up_conv7 = TransposeSingleConv(8, 4, kernel_size=kernel_size, scale_factor=(2, 1, 1),
                                             padding=(2,1,1), order ='cgr', num_groups=2)
        self.conv_final = SingleConv(4, 1, kernel_size=1, order='cgr', num_groups=1, padding=(0,0,0),stride=(1,1,1))

        self.up_sampling = InterpolateUpsampling()

    def forward(self, x):
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_conv3(x)
        x = self.up_conv4(x)
        x = self.up_conv5(x)
        x = self.up_conv6(x)
        x = self.up_conv7(x)
        x = self.conv_final(x) #4,4,32
        torch.cuda.empty_cache()

        return x


class Discriminator(nn.Module):
    def __init__(self, kernel_size = (4,3,3)):
        super(Discriminator, self).__init__()
        self.conv1 = SingleConv(48, 16, kernel_size=(1,1,1), order='cgr', num_groups=1, padding=(0,0,0),stride=(1,1,1))
        self.conv2 = SingleConv(16, 4, kernel_size=(1,1,1), order='cgr', num_groups=1, padding=(0,0,0),stride=(1,1,1))
        self.linear1 = nn.Linear(48*8*4*4,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128, 1)
        self.pooling = nn.AvgPool3d(kernel_size=(1,8,8))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(-1, 48*8*4*4)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        output = self.sigmoid(x)

        return output
