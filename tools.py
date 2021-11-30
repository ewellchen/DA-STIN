#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct the tools. """


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import importlib
import logging
import os
import shutil
import sys
import numpy as np


loggers = {}
def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        # Logging to file
        # file_handler = logging.FileHandler('log.txt', mode='a')
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

        loggers[name] = logger

        return logger

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('Folder have been made!')

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def save_checkpoint(models, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        logger: log
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    for index in range(len(models)):
        last_file_path =  checkpoint_dir + '/' + 'last_checkpoint%d.pytorch'%(index)
        log_info(f"Saving last checkpoint to '{last_file_path}'")
        torch.save(models[index], last_file_path)
        if is_best:
            best_file_path = checkpoint_dir + '/' + 'best_checkpoint%d.pytorch'%(index)
            log_info(f"Saving best checkpoint to '{best_file_path}'")
            shutil.copyfile(last_file_path, best_file_path)

def load_checkpoint_test(checkpoint_dir, models):

    if not os.path.exists(checkpoint_dir):
        raise IOError(f"Checkpoint '{checkpoint_dir}' does not exist")
    models_resume = []
    for index in range(len(models)):
        file_path = os.path.join(checkpoint_dir, 'best_checkpoint%d.pytorch'%index)
        model = torch.load(file_path, map_location='cpu')
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        models_resume.append(model)

    return models_resume

def load_checkpoint_train(checkpoint_dir, models):
    if not os.path.exists(checkpoint_dir):
        raise IOError(f"Checkpoint '{checkpoint_dir}' does not exist")
    models_resume = []
    for index in range(len(models)):
        file_path = os.path.join(checkpoint_dir, 'best_checkpoint%d.pytorch'%index)
        model = torch.load(file_path, map_location='cpu')
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        models_resume.append(model)

    return models_resume

class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

    def zero(self):
        self.count += 0
        self.sum += 0
        self.avg = 0

class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError

class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))

def log_lr(optimizer,writer,idx):
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('learning_rate', lr, idx)

def log_stats(phase, value, writer,idx):
    tag_value = {
        f'{phase}_value': value,
    }
    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, idx)

def log_params(model, writer,idx,logger):
    logger.info('Logging model parameters and gradients')
    for name, value in model.named_parameters():
        writer.add_histogram(name, value.data.cpu().numpy(), idx)
        writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), idx)

def log_images(spad, depth, predict_depth, writer,idx,  prefix=''):

    # soft_argmax = torch.squeeze(soft_argmax, 1)
    inputs_map = {
        'inputs': spad,
        'targets': depth,
        'predictions': predict_depth
    }
    img_sources = {}
    for name, batch in inputs_map.items():
        if isinstance(batch, list) or isinstance(batch, tuple):
            for i, b in enumerate(batch):
                img_sources[f'{name}{i}'] = b.data.cpu().numpy()
        else:
            img_sources[name] = batch.data.cpu().numpy()

    for name, batch in img_sources.items():
        for tag, image in DefaultTensorboardFormatter()(name, batch):
            writer.add_image(prefix + tag, image, idx, dataformats='CHW')