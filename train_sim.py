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


def train_one_epoch(models, optimizers, batch_size, device, train_loader, epoch,
                    logger, writer, checkpoint_dir, result_dir, test_loader, is_adver=False):

    def _get_lambda(epoch, max_epoch):
        p = (epoch) / max_epoch
        return 2. / (1 + np.exp(-10. * p)) - 1
    def _transtodepth(prediction):
        pre = torch.squeeze(prediction, 1)
        smax = torch.nn.Softmax2d()
        weights = torch.linspace(0, 1, steps=pre.size()[1]).unsqueeze(1).unsqueeze(1).type(
            torch.cuda.FloatTensor)
        weighted_smax = weights * smax(pre)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)
        return soft_argmax

    index = epoch * int(train_loader.dataset.src_size/batch_size)
    loss_reconstructor = RunningAverage()
    loss_discriminator = RunningAverage()
    loss_sum = RunningAverage()
    total = len(train_loader.dataset)
    global best_test_score
    global test_idx
    global test_score

    for model in models:
        model.train()
    F = models[0].to(device)
    R = models[1].to(device)
    D = models[2]
    for opt in optimizers:
        opt.zero_grad()
    opt_F = optimizers[0]
    opt_R = optimizers[1]
    opt_D = optimizers[2]
    if is_adver:
        D = D.to(device)

    for batch_idx, data in enumerate(train_loader):
        src_spad = data['src_spad'].to(device)
        src_depth = data['src_depth'].to(device)
        spad = src_spad
        opt_R.zero_grad()
        opt_F.zero_grad()
        features = F(spad)
        predict_src = R(features)
        # pre_depth = torch.argmax(predict_src, dim=2, keepdim=False)
        pre_depth = _transtodepth(predict_src)
        loss_re = CrossEntropyLoss()(torch.squeeze(predict_src, dim=1), torch.squeeze(src_depth, dim=1)) \
                  + 0.005 * criterion_TV(pre_depth)
        loss_re.backward(retain_graph=False)
        opt_R.step()
        opt_F.step()
        loss_reconstructor.update(loss_re.item())

        del features
        torch.cuda.empty_cache()

        if batch_idx % 500 ==0:
            with torch.no_grad():
                test_sc,_ = test([F, R], test_loader, device, test_idx, logger, writer, result_dir)
                test_score.append(test_sc)
                test_idx = test_idx +1
                print(test_score)
                if test_sc < best_test_score:
                    best_test_score = test_sc
                    # save_checkpoint([F,R,D], True, checkpoint_dir)

        if batch_idx % 2 == 0:
            begin = batch_idx * len(data['src_spad'])
            percent = int(100. * batch_idx / len(train_loader))
            # logger.info(f"Getting {train_set.src_size} source samples for training")
            # logger.info(f"Getting {train_set.tar_size} target samples for training")
            logger.info(f"Train Epoch: {epoch} [{begin}/{total} ({percent}%)]\t Loss_re: {loss_reconstructor.avg}")
            # logger.info(f"Train Epoch: {epoch} [{begin}/{total} ({percent}%)]\t Loss_dis4: {loss_discriminator.avg}")
            # logger.info(f"Train Epoch: {epoch} [{begin}/{total} ({percent}%)]\t Loss_total: {loss_sum.avg}")
            log_lr(opt_R, writer, index + batch_idx)
            # log_params(opt_R, writer, index,logger)
            log_stats('reconstructor_loss', loss_reconstructor.avg, writer, index + batch_idx)
            # log_stats('discriminator_loss', loss_discriminator.avg, writer, index + batch_idx)
            # log_stats('total_loss', loss_sum.avg, writer, index + batch_idx)
            log_images(spad[:batch_size], src_depth, pre_depth, writer, index + batch_idx)
            save_checkpoint([F, R, D], False, checkpoint_dir)
            loss_reconstructor.zero(), loss_discriminator.zero(), loss_sum.zero()

def test(models, testloader, device, epoch, logger, writer,result_dir):
    def log_info(message):
        if logger is not None:
            logger.info(message)
    for mod in models:
        mod.eval()
    F = models[0].to(device)
    R = models[1].to(device)
    Metric = []
    index = 0
    mean_metric = 0
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
            tgt_spad = data['tgt_spad'].to(device)
            tgt_depth = torch.squeeze(data['tgt_depth'].to(device))
            predict_src = R(F(tgt_spad))
            # pre_depth = torch.argmax(predict_src, dim=2, keepdim=False)

            pre_depth = _transtodepth(predict_src)
            K = 80* 1e-12 * 3 * 1e8 * 1023 /2
            pre_depth = torch.squeeze(K * pre_depth.float())
            metric_mse = torch.sqrt(torch.pow(pre_depth- tgt_depth,2).sum()/(72*88)).cpu().numpy() #mse
            mean_metric = mean_metric + metric_mse

            metric_abs = (torch.abs(pre_depth - tgt_depth)/tgt_depth).sum().cpu().numpy() / (72 * 88)  # Abs rel
            thr = 1.01
            num = 0
            for i in range(72):
                for j in range(88):
                    a = torch.max(pre_depth[i,j]/tgt_depth[i,j],tgt_depth[i,j]/pre_depth[i,j])
                    if a < thr:
                        num = num + 1
            metric_accuracy_1 = num /( 72*88)

            thr = 1.01*1.01
            num = 0
            for i in range(72):
                for j in range(88):
                    a = torch.max(pre_depth[i,j]/tgt_depth[i,j],tgt_depth[i,j]/pre_depth[i,j])
                    if a < thr:
                        num = num + 1
            metric_accuracy_2 = num /( 72*88)

            thr = 1.01*1.01*1.01
            num = 0
            for i in range(72):
                for j in range(88):
                    a = torch.max(pre_depth[i,j]/tgt_depth[i,j],tgt_depth[i,j]/pre_depth[i,j])
                    if a < thr:
                        num = num + 1
            metric_accuracy_3 = num /( 72*88)

            print('RMSE:',metric_mse,'ABS:',metric_abs,'a_1:',metric_accuracy_1,'a_2:',metric_accuracy_2,'a_3:',metric_accuracy_3)

            if not os.path.exists(result_dir):
                log_info(
                    f"Checkpoint directory does not exists. Creating {result_dir}")
                os.mkdir(result_dir)

            from scipy import io
            io.savemat(result_dir + '/depth_%d_%d_%d.mat' % (
            index, data['tgt_photon'], data['tgt_photon'].float() / data['tgt_sbr']),
                       {'depth': pre_depth.cpu().numpy()})
            mpimg.imsave(result_dir + '/depth_%d_%d_%d.jpg'%(index,data['tgt_photon'],data['tgt_photon'].float()/data['tgt_sbr']),
                         torch.squeeze(pre_depth).cpu())#,cmap = 'gray')
            index = index+1
            torch.cuda.empty_cache()
        mean_metric = mean_metric / 8
        return mean_metric, Metric

    # print('Accuracy:%f %%' % (100*correct/total))

def trainer(config):
    logger = get_logger('Training')

    #load dataset
    train_set = Train_set(root_dir_src=config['src_train_set_path'], root_dir_tgt=config['tgt_train_set_path'],
                          phase='train', shape=32)
    test_set = Test_set(root_dir_tgt=config['test_set_small_path'], phase='test', img_size=config['test_size_small'])
    batch_size_train = config['train_config']['batch_size']
    batch_size_test = config['test_config']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    logger.info(f"Getting {train_set.src_size} samples for source training")
    logger.info(f"Getting {train_set.tgt_size} samples for target training")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)
    logger.info(f"Getting {test_set.tgt_size} samples for testing")

    #define models and optimizers, put the model on GPUs
    device = config['train_config']['device']
    F = Feature_extractor(kernel_s = 3, kernel_t = 7)
    R = Reconstructor(kernel_size=(6,3,3))
    D = Discriminator(kernel_size=(4,3,3))
    ##############################################################################################################
    resume = config.get('resume', True)
    if resume:
        checkpoint_dir = config.get('checkpoint_dir')
        [F, R] = load_checkpoint_train(checkpoint_dir=checkpoint_dir, models=[F, R])
    ##############################################################################################################
    learning_rate = config['train_config']['learning_rate']
    optimizer_F = torch.optim.Adam(F.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_R = torch.optim.Adam(R.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=learning_rate, weight_decay=1e-5)
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        F = nn.DataParallel(F)
        R = nn.DataParallel(R)
        D = nn.DataParallel(D)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
    logger.info(f"Sending the model to {device}")
    F = F.to(device)
    R = R.to(device)
    D = D.to(device)
    logger.info(f'Number of F learnable params {get_number_of_learnable_parameters(F)}')
    logger.info(f'Number of R learnable params {get_number_of_learnable_parameters(R)}')
    logger.info(f'Number of D learnable params {get_number_of_learnable_parameters(D)}')

    # tensorboard
    checkpoint_dir = config['checkpoint_dir']
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
    data_1 = torch.rand(1, 1, 1024, 32, 32).to(device)
    writer.add_graph(F, data_1)

    for epoch in range(config['train_config']['epoch']):
        result_dir = config['result_dir_small']
        train_one_epoch([F,R,D], [optimizer_F,optimizer_R,optimizer_D], batch_size_train, device, train_loader, epoch,
                         logger, writer, checkpoint_dir, result_dir, test_loader,is_adver=False)
        save_checkpoint([F,R,D], False, checkpoint_dir)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    config = CONFIG
    best_test_score = 1
    test_idx = 0
    test_score = []
    trainer(config)
