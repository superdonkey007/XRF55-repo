import logging

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class XRFMmwaveDataset(Dataset):
    def __init__(self, file_path='/root/autodl-tmp/dataset/XRF_dataset/', is_train=True, scene='dml'):
        super().__init__()
        self.word_list = np.load('./word2vec/bert_new_sentence_large_uncased.npy')
        self.file_path = file_path
        self.is_train = is_train
        self.scene = scene
        split = 'train' if self.is_train else 'val'
        self.index_file = f'{self.file_path}{self.scene}_{split}.txt'

        with open(self.index_file, 'r') as file:
            lines = file.readlines()

        self.data = {
            'file_name': [],
            'label': [],
        }
        self.path = self.file_path

        for line in lines:
            parts = line.strip().split(',')
            self.data['file_name'].append(parts[0])
            self.data['label'].append(int(parts[2]) - 1)

        log.info('load mmWave-only XRF dataset')

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        vector = self.word_list[label]
        mmwave_data = load_mmwave(file_name, self.is_train, path=self.path)
        return mmwave_data, label, vector


def load_mmwave(filename, is_train, path='/root/autodl-tmp/dataset/XRF_dataset/'):
    split_dir = 'train_data/' if is_train else 'test_data/'
    mmwave_data = np.load(path + split_dir + 'mmWave/' + filename + '.npy')
    return torch.from_numpy(mmwave_data).float()
