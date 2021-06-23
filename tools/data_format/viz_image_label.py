# -*- coding: utf-8 -*-
# @Time : 2021/6/21 11:46 下午
# @Author : sunqiang
# @Email : wayne_lau@aliyun.com
# @File : viz_image_label.py
# @Project : Tools
import os

import numpy as np
import cv2


def drew_box(image, label):
    for i in range(len(label)):
        cv2.rectangle(image, label[i,0:2], label[i,2:4], (0, 255, 0))
        title = "class: {}".format(label[i,-1])
        position = (max(label[i,0], 15), max(label[i,1], 15))
        cv2.putText(image, title, position, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow('viz',image)
    cv2.waitKey(1000)

def show_segment_pic(image,label):
    if image.shape != label.shape:
        raise ValueError("尺寸不匹配")
    show_pic = np.zeros((image.shape[0]*2,image.shape[1],3))
    show_pic[:image.shape,:,:] = image
    show_pic[image.shape:,:,:]  = label
    cv2.imshow('viz',image)
    cv2.waitKey(1000)


def viz_image_label(image_dir, label_dir):
    image_list = [image_dir + os.sep + file for file in sorted(os.listdir(image_dir))]
    label_list = [label_dir + os.sep + file for file in sorted(os.listdir(label_dir))]

    for image, label in zip(image_list, label_list):
        image = cv2.imread(image)
        if label.split('.')[1] == 'npy':
            label = np.load(label).astype(np.int32)
            drew_box(image,label)
        elif label.split('.')[1] == 'png':
            label = cv2.imread(label)
            show_segment_pic(image,label)



if __name__ == '__main__':
    image_dir = '/Users/sunqiang/data/trainval/VOC2007/JPEGImages/'
    label_dir = '/Users/sunqiang/data/trainval/VOC2007/SegmentationClass/'
    viz_image_label(image_dir,label_dir)