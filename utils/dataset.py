# -*- coding: utf-8 -*-
# @Time : 2021/6/23 10:37 下午
# @Author : sunqiang
# @Email : wayne_lau@aliyun.com
# @File : dataset.py
# @Project : DL_Hub

import torch
import cv2
import numpy as np

import glob

from torch.utils.data import DataLoader


class Dataset:
    def __init__(self, img_dir, label_dir, mode):
        '''
        :param img_dir:
        :param label_dir:
        :param mode: 训练模式，segment or detect
        '''
        self.image_list = sorted(glob.glob(img_dir + '/*'))
        self.label_list = sorted(glob.glob(label_dir + '/*'))
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        if self.mode == 'segment':
            label = cv2.imread(self.label_list[index])
        else:
            label = np.load(self.label_list[index])

        return {"image": image, 'label': label}


def dataloader(img_dir, label_dir, mode, batch_size=1, shuffer=True, num_works=4):
    '''
    :param img_dir:
    :param label_dir:
    :param mode: 训练模式，segment or detect
    :param batch_size:
    :param shuffer:
    :param num_works:
    :return:
    '''
    dataset = Dataset(img_dir, label_dir, mode)
    dataloader_ = DataLoader(dataset, batch_size=batch_size, shuffle=shuffer, num_workers=num_works)
    return dataloader_


if __name__ == '__main__':
    from tools.data_format.viz_image_label import drew_box

    img_dir = '../data/detection/trainval/img'
    label_dir = '../data/detection/trainval/label'
    for sample in dataloader(img_dir, label_dir, 'detect'):
        image = sample['image'][0].numpy()
        label = sample['label'][0].numpy()
        drew_box(image, label)
