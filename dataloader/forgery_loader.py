from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.image_util import read_text_lines, resize_max_res


class ForgeryDataset(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 dataset_name='casia',
                 processing_res=768,
                 transform=None):
        super(ForgeryDataset, self).__init__()

        self.transform = transform
        self.processing_res = processing_res
        self.samples = []
        lines = read_text_lines(train_datalist)

        for line in lines:
            sample = dict()
            if dataset_name == 'coco':
                sample['original'] = os.path.join(data_dir, 'fake', line)
                sample['gt'] = os.path.join(data_dir, 'mask', line.split('.')[0] + '.png')
            elif dataset_name == 'casia':
                sample['original'] = os.path.join(data_dir, 'Tp', line)
                sample['gt'] = os.path.join(data_dir, 'Gt', line.split('.')[0] + '.png')
            else:
                assert False, 'dataset name setting error %s' % dataset_name

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        original = Image.open(sample_path['original'])
        original = resize_max_res(original, self.processing_res)
        sample['original'] = np.array(original.convert('RGB')).astype(np.float32)
        gt = Image.open(sample_path['gt'])
        gt = resize_max_res(gt, self.processing_res)
        tmp = np.array(gt).astype(np.float32)
        if len(tmp.shape) == 2:
            tmp = np.expand_dims(tmp, 2)
        else:
            tmp = tmp[:, :, 0:1]
        sample['gt'] = np.repeat(tmp, 3, 2)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
