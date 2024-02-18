from __future__ import division
import torch
import numpy as np
import torch.nn.functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor"""
    def __call__(self, sample):
        original = np.transpose(sample['original'], (2, 0, 1))
        sample['original'] = torch.from_numpy(original) / 255.
        if 'gt' in sample.keys():
            gt = np.transpose(sample['gt'], (2, 0, 1))
            sample['gt'] = torch.from_numpy(gt) / 255.

        return sample
