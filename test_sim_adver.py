#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import torch
import numpy as np
import os
import scipy

import matplotlib.image as mpimg
import torch.nn as nn
from tensorboardX import SummaryWriter
from scipy.sparse import csc_matrix
from configuration import CONFIG
from tools import mkdir, get_logger, get_number_of_learnable_parameters, save_checkpoint, load_checkpoint_test, \
    RunningAverage,log_lr,log_images,log_params,log_stats, load_checkpoint_train
from losses import criterion_TV
from torch.nn import MSELoss, SmoothL1Loss, L1Loss, CrossEntropyLoss, BCELoss
from model import Feature_extractor, Reconstructor, Discriminator
from torch.utils.data import Dataset
from My_dataset import Train_set, Test_set
from torchsummary import summary


def test_large(models, testloader, device, epoch, logger, writer,result_dir):
    def log_info(message):
        if logger is not None:
            logger.info(message)
    for mod in models:
        mod.eval()
    F = models[0].to(device)
    R = models[1].to(device)
    Metric = []
    index = 0
    def _transtodepth(prediction):
        pre = torch.squeeze(prediction, 1)
        smax = torch.nn.Softmax2d()
        weights = torch.linspace(0, 1, steps=pre.size()[1]).unsqueeze(1).unsqueeze(1).type(
            torch.cuda.FloatTensor)
        weighted_smax = weights * smax(pre)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)
        return soft_argmax
    with torch.no_grad():
        for data in testloader:

            pre_depth = torch.zeros([512,512]).to(device)
            spad = data['tgt_spad'].to(device)
            spad1 = torch.zeros([1,1,1024,544,544]).to(device)
            spad1[:,:,:,16:528,16:528] = spad
            del spad
            torch.cuda.empty_cache()

            for i in range(0,4):
                for j in range(0,4):
                    chuck = spad1[:,:,:,(128 * i):(128 * i+160),(128 * j):(128 * j+160)]
                    predict_src = R(F(chuck))
                    depth = torch.squeeze(_transtodepth(predict_src))
                    pre_depth[128 * i:(128 * i+128),(128 * j):(128 * j+128)] = depth[16:144,16:144]
                    del chuck, predict_src, depth
                    torch.cuda.empty_cache()

            tgt_depth = data['tgt_depth'].to(device)
            K = 80* 1e-12 * 3 * 1e8 * 1023 /2
            pre_depth = K * pre_depth
            metric = torch.sqrt(torch.pow(pre_depth.float() - tgt_depth, 2).sum() / (512 * 512)).cpu().numpy()
            Metric.append(metric)
            # log_images(tgt_spad, tgt_depth, pre_depth, writer, epoch, prefix='test-')
            if not os.path.exists(result_dir):
                log_info(
                    f"Checkpoint directory does not exists. Creating {result_dir}")
                os.mkdir(result_dir)
            # for i in range(pre_depth.size()[0]):
            from scipy import io
            io.savemat(result_dir + '/depth_%d_%d_%d.mat'%(index,data['tgt_photon'],data['tgt_photon'].float()/data['tgt_sbr']),
                       {'depth': pre_depth.cpu().numpy()})
            mpimg.imsave(result_dir + '/depth_%d_%d_%d.jpg'%(index,data['tgt_photon'],data['tgt_photon'].float()/data['tgt_sbr']),
                             torch.squeeze(pre_depth).cpu())#,cmap = 'gray')
            index = index+1
            torch.cuda.empty_cache()
        logger.info("metrics:" + str(np.array(Metric)))
        mean_metric = np.mean(np.array(Metric))
        log_stats('test_mse', mean_metric, writer, epoch)
        return mean_metric, Metric


def tester(config):
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    logger = get_logger('Testing')
    test_set = Test_set(root_dir_tgt=config['test_set_large_path'], phase='test', img_size=config['test_size_large'])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    batch_size_test = config['test_config']['batch_size']
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)
    logger.info(f"Getting {test_set.tgt_size} samples for testing")
    # parameter setting
    checkpoint_dir = config['checkpoint_dir']
    result_dir = config['result_dir_large']
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
    device_test = config['test_config']['device']
    F = Feature_extractor(kernel_s=3, kernel_t=7)
    R = Reconstructor(kernel_size=(6, 3, 3))
    [F, R] = load_checkpoint_test(checkpoint_dir=checkpoint_dir, models=[F, R])
    summary(F.cuda(),(1,1024,32,32))
    summary(R.cuda(), (48, 4, 32, 32))
    with torch.no_grad():
        test_sc, metric = test_large([F, R], test_loader, device_test, 1, logger, writer, result_dir)
        print('test_score: %f' % test_sc)
        fo = open("result.txt", "a")
        fo.write("\n" + "\n" + result_dir)
        fo.write("\n" + 'test_score: %f' % test_sc)
        fo.write("\n" + "metrics:" + str(np.array(metric)))
        fo.close()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    config = CONFIG
    best_test_score = 1
    test_idx = 0
    test_score = []
    tester(config)