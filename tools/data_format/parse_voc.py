# -*- coding: utf-8 -*-
# ----------------------------
# !  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：刘恩甫
#   完成日期：2019-x-x
# -----------------------------
import shutil

import cv2
import os
import xml.dom.minidom
import numpy as np
import xml.dom.minidom as xmldom


def get_class_list(annotation_dir):
    annotation_names = [os.path.join(annotation_dir, i) for i in os.listdir(annotation_dir)]

    labels = list()
    for names in annotation_names:
        xmlfilepath = names
        domobj = xmldom.parse(xmlfilepath)
        # 得到元素对象
        elementobj = domobj.documentElement
        # 获得子标签
        subElementObj = elementobj.getElementsByTagName("object")
        for s in subElementObj:
            label = s.getElementsByTagName("name")[0].firstChild.data
            # print(label)
            if label not in labels:
                labels.append(label)
    print("labels: ", labels)
    return labels

def transform_voc_npy(img_dir,anno_dir,label_save_dir):
    labels = get_class_list(annotation_path=anno_dir)

    imagelist = os.listdir(img_dir)

    char2id = {str(v): index for index, v in enumerate(labels)}
    if not os.dir.exists(label_save_dir):
        os.makedirs(label_save_dir)

    for image in imagelist:
        print(image)
        image_pre, ext = os.path.splitext(image)
        img_file = img_dir + image
        img = cv2.imread(img_file)
        xml_file = anno_dir + image_pre + '.xml'
        DOMTree = xml.dom.minidom.parse(xml_file)
        collection = DOMTree.documentElement
        objects = collection.getElementsByTagName("object")

        all_labels = []
        for object in objects:
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = xmin.childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = ymin.childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = xmax.childNodes[0].data
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = ymax.childNodes[0].data
            xmin = int(xmin_data)
            xmax = int(xmax_data)
            ymin = int(ymin_data)
            ymax = int(ymax_data)
            name_obj = object.getElementsByTagName('name')[0]
            name_char = name_obj.childNodes[0].data
            name_id = char2id[name_char]
            all_labels.append([xmin, ymin, xmax, ymax, name_id])

        print(all_labels)
        np.save(label_save_dir + image.split('.')[0] + '.npy', all_labels)


def select_segment_data(image_dir,segment_dir,save_dir):
    label_list = os.listdir(segment_dir)
    for label in label_list:
        image = image_dir+os.sep+label.split('.')[0]+'.jpg'
        shutil.copy(image,save_dir+os.sep+label.split('.')[0]+'.jpg')


if __name__ == '__main__':
    from imgFileOpterator import Img_processing

    img_dir = '/Users/sunqiang/data/test/VOC2007/JPEGImages/'
    anno_dir = '/Users/sunqiang/data/test/VOC2007/Annotations/'

    label_save_dir = '/Users/sunqiang/data/test/VOC2007/detect_label/'
    # transform_voc_npy(img_dir,anno_dir,label_save_dir)

    img_dir = '/Users/sunqiang/data/test/VOC2007/JPEGImages/'
    segment_dir = '/Users/sunqiang/data/test/VOC2007/SegmentationClass/'
    save_dir = '/Users/sunqiang/data/test/VOC2007/segment/img'
    select_segment_data(img_dir,segment_dir,save_dir)

