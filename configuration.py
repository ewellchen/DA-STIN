#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Default configurations of model configuration, training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from typing import Dict

CONFIG = {
    'is_train': True,

    'src_train_set_path': './train_data_source',

    'tgt_train_set_path': './train_data_target',

    'test_set_small_path': './test_data/low_resolution/P2-100',

    'test_set_large_path': './test_data/high_resolution/P2-100',

    'test_size_small': [72,88],

    'test_size_large': [512, 512],

    'checkpoint_dir': './checkpoint',

    'result_dir_small': './results/STIN-small',

    'result_dir_large': './results/STIN-large',

    'resume': True,

    'train_config': {'epoch': 5,
                     'batch_size': 4,
                     'device': 'cuda:0',
                     'learning_rate': 0.0005,},

    'train_config_adv': {'epoch': 5,
                     'batch_size': 2,
                     'device': 'cuda:0',
                     'learning_rate': 0.0005, },

    'test_config': {'batch_size': 1,
                    'device': 'cuda:0', },
}

CONFIG_NONLOCAL = {

    'test_set_path': './test_data/low_resolution/P2-100',

    'test_size': [72,88],

    'result_dir': './result/non-local-small',

    'test_config': {'batch_size': 1,
                    'device': 'cuda:0', },
}

CONFIG_UNETPP = {

    'test_set_path': './test_data/low_resolution/P2-100',

    'test_size': [72,88],

    'result_dir': './result/unetpp-small',

    'test_config': {'batch_size': 1,
                    'device': 'cuda:0', },
}
