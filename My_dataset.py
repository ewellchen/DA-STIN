import torch
import numpy as np
import os
import scipy
from torch.utils.data import Dataset
from scipy.sparse import csc_matrix

class Train_set(Dataset):
    def __init__(self, root_dir_src, root_dir_tgt, phase, shape = 32):
        assert os.path.isdir(root_dir_src), f'{root_dir_src} is not a directory'
        assert os.path.isdir(root_dir_tgt), f'{root_dir_tgt} is not a directory'
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.src_files = []
        self.shape = shape
        # if self.phase == 'train':
        for root, dirs, src_files in os.walk(root_dir_src):
            for file in src_files:
                    self.src_files.append(os.path.join(root, file))
        self.src_size = len(self.src_files)
        self.tgt_files = []
        for root, dirs, tgt_files in os.walk(root_dir_tgt):
            for file in tgt_files:
                self.tgt_files.append(os.path.join(root, file))
        self.tgt_size = len(self.tgt_files)


    def __len__(self):
        return self.src_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        try:
            sample = self._getitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self._getitem(idx)
        return sample

    def _getitem(self, idx):
        src_spad = np.asarray(csc_matrix.todense(scipy.io.loadmat(self.src_files[idx])['spad'])).astype(np.float32) \
            .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        src_spad = np.transpose(src_spad, (0, 3, 2, 1))  # (1, 1024, 64, 64)
        tar_idx = idx
        if idx > self.tgt_size:
            tar_idx = idx % self.tgt_size
        tgt_spad = np.asarray(csc_matrix.todense(scipy.io.loadmat(self.tgt_files[tar_idx])['spad'])).astype(np.float32) \
            .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        tgt_spad = np.transpose(tgt_spad, (0, 3, 2, 1))  # (1, 1024, 64, 64)
        src_depth = (np.asarray(scipy.io.loadmat(self.src_files[idx])['bin']).astype(np.int64)
                     .reshape([1, 64, 64]) - 1)  # (1, 64, 64)
        # tgt_depth = (np.asarray(scipy.io.loadmat(self.tgt_files[idx])['bin']).astype(np.int64)
        #              .reshape([1, 64, 64]) - 1)  # (1, 64, 64)

        # rates = np.asarray(scipy.io.loadmat(self.files[idx])['rates']).astype(np.float32) \
        #     .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        # rates = np.transpose(rates, (0, 3, 1, 2))  # (1, 1024, 64, 64)
        # rates = rates / np.sum(rates, axis=1)[None, :, :, :]
        h, w = src_spad.shape[2:]
        new_h = self.shape
        new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        src_spad = src_spad[:, :, top: top + new_h, left: left + new_w]
        tgt_spad = tgt_spad[:, :, top: top + new_h, left: left + new_w]
        src_depth = src_depth[:, top: top + new_h, left: left + new_w]

        src_spad, tgt_spad, src_depth = torch.from_numpy(src_spad), torch.from_numpy(tgt_spad), \
                                                   torch.from_numpy(src_depth)

        sample = {'src_spad': src_spad, 'tgt_spad': tgt_spad, 'src_depth': src_depth}

        return sample

class Test_set(Dataset):
    def __init__(self, root_dir_tgt, phase, img_size):
        assert os.path.isdir(root_dir_tgt), f'{root_dir_tgt} is not a directory'
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.src_files = []
        # if self.phase == 'train':
        self.tgt_files = []
        for root, dirs, tgt_files in os.walk(root_dir_tgt):
            for file in tgt_files:
                self.tgt_files.append(os.path.join(root, file))
        self.tgt_size = len(self.tgt_files)
        self.img_size = img_size


    def __len__(self):
        return self.tgt_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        try:
            sample = self._getitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self._getitem(idx)
        return sample

    def _getitem(self, idx):
        tgt_spad = np.asarray(csc_matrix.todense(scipy.io.loadmat(self.tgt_files[idx])['spad'])).astype(np.float32) \
            .reshape([1, self.img_size[1], self.img_size[0], -1])  # (1, 64, 64, 1024)
        tgt_spad = np.transpose(tgt_spad, (0, 3, 2, 1))  # (1, 1024, 64, 64)
        tgt_depth = (np.asarray(scipy.io.loadmat(self.tgt_files[idx])['depth']).astype(np.float32)
                     .reshape([1, self.img_size[0], self.img_size[1]]))  # (1, 64, 64)

        # rates = np.asarray(scipy.io.loadmat(self.files[idx])['rates']).astype(np.float32) \
        #     .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        # rates = np.transpose(rates, (0, 3, 1, 2))  # (1, 1024, 64, 64)
        # rates = rates / np.sum(rates, axis=1)[None, :, :, :]
        tgt_photon = scipy.io.loadmat(self.tgt_files[idx])['mean_signal_photons'][0].astype(np.int64)
        tgt_sbr = scipy.io.loadmat(self.tgt_files[idx])['SBR'][0].astype(np.float32)
        tgt_spad, tgt_depth = torch.from_numpy(tgt_spad), torch.from_numpy(tgt_depth)

        sample = {'tgt_spad': tgt_spad,'tgt_depth': tgt_depth, 'tgt_photon': tgt_photon,'tgt_sbr': tgt_sbr }

        return sample



class Train_real(Dataset):
    def __init__(self, root_dir_src, root_dir_tgt, phase, shape = 32):
        assert os.path.isdir(root_dir_src), f'{root_dir_src} is not a directory'
        assert os.path.isdir(root_dir_tgt), f'{root_dir_tgt} is not a directory'
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.src_files = []
        self.shape = shape
        # if self.phase == 'train':
        for root, dirs, src_files in os.walk(root_dir_src):
            for file in src_files:
                    self.src_files.append(os.path.join(root, file))
        self.src_size = len(self.src_files)
        self.tgt_files = []
        for root, dirs, tgt_files in os.walk(root_dir_tgt):
            for file in tgt_files:
                self.tgt_files.append(os.path.join(root, file))
        self.tgt_size = len(self.tgt_files)


    def __len__(self):
        return self.src_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        try:
            sample = self._getitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self._getitem(idx)
        return sample

    def _getitem(self, idx):
        src_spad = np.asarray(csc_matrix.todense(scipy.io.loadmat(self.src_files[idx])['spad'])).astype(np.float32) \
            .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        src_spad = np.transpose(src_spad, (0, 3, 2, 1))  # (1, 1024, 64, 64)
        tar_idx = idx
        if idx > self.tgt_size:
            tar_idx = idx % self.tgt_size
        tgt_spad = np.expand_dims(np.asarray(scipy.io.loadmat(self.tgt_files[tar_idx])['spad']).astype(np.float32),0)
        #tgt_spad = np.transpose(tgt_spad, (0, 1, 3, 2))
        src_depth = (np.asarray(scipy.io.loadmat(self.src_files[idx])['bin']).astype(np.int64)
                     .reshape([1, 64, 64]) - 1)  # (1, 64, 64)
        # tgt_depth = (np.asarray(scipy.io.loadmat(self.tgt_files[idx])['bin']).astype(np.int64)
        #              .reshape([1, 64, 64]) - 1)  # (1, 64, 64)

        # rates = np.asarray(scipy.io.loadmat(self.files[idx])['rates']).astype(np.float32) \
        #     .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        # rates = np.transpose(rates, (0, 3, 1, 2))  # (1, 1024, 64, 64)
        # rates = rates / np.sum(rates, axis=1)[None, :, :, :]
        h, w = src_spad.shape[2:]
        h_t, w_t = tgt_spad.shape[2:]
        new_h = self.shape
        new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        top_t = np.random.randint(0, h_t - new_h)
        left_t = np.random.randint(0, w_t - new_w)


        src_spad = src_spad[:, :, top: top + new_h, left: left + new_w]
        tgt_spad = tgt_spad[:, :, top_t: top_t + new_h, left_t: left_t + new_w]
        src_depth = src_depth[:, top: top + new_h, left: left + new_w]

        src_spad, tgt_spad, src_depth = torch.from_numpy(src_spad), torch.from_numpy(tgt_spad), \
                                                   torch.from_numpy(src_depth)

        sample = {'src_spad': src_spad, 'tgt_spad': tgt_spad, 'src_depth': src_depth}

        return sample

class Test_real(Dataset):
    def __init__(self, root_dir_tgt, phase, img_size):
        assert os.path.isdir(root_dir_tgt), f'{root_dir_tgt} is not a directory'
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.src_files = []
        # if self.phase == 'train':
        self.tgt_files = []
        for root, dirs, tgt_files in os.walk(root_dir_tgt):
            for file in tgt_files:
                self.tgt_files.append(os.path.join(root, file))
        self.tgt_size = len(self.tgt_files)
        self.img_size = img_size


    def __len__(self):
        return self.tgt_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration
        try:
            sample = self._getitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self._getitem(idx)
        return sample

    def _getitem(self, idx):
        tgt_spad = np.expand_dims(np.asarray(scipy.io.loadmat(self.tgt_files[idx])['spad']).astype(np.float32),0)
        #tgt_spad = np.transpose(tgt_spad, (0, 1, 3, 2))
        # rates = np.asarray(scipy.io.loadmat(self.files[idx])['rates']).astype(np.float32) \
        #     .reshape([1, 64, 64, -1])  # (1, 64, 64, 1024)
        # rates = np.transpose(rates, (0, 3, 1, 2))  # (1, 1024, 64, 64)
        # rates = rates / np.sum(rates, axis=1)[None, :, :, :]
        tgt_spad = torch.from_numpy(tgt_spad)

        sample = {'tgt_spad': tgt_spad}

        return sample
