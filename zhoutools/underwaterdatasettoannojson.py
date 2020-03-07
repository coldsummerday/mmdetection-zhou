#!/usr/bin/env python
import os.path as osp
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import argparse
import pickle as pkl
"""
 Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

"""

classes = ["holothurian","echinus","scallop","starfish"]
cat2label = {cat: i + 1 for i, cat in enumerate(classes)}
def convert(data_root):
    img_infos = []
    filenames = os.listdir(osp.join(data_root, "train", "image"))
    img_ids = [id.split('.')[0] for id in filenames]
    for img_id in img_ids:
        filename = osp.join(data_root, 'train', "image",
                            '{}.jpg'.format(img_id))
        # 需要打开才知道图片宽高,需要做一个annotation.json快速读取
        img_cv = cv2.imread(filename)

        img_info = dict(id=img_id, filename=filename, width=img_cv.shape[1], height=img_cv.shape[0])

        del img_cv

        #打开xml 获取object bbox信息
        xml_path = osp.join(data_root, 'train', "box",
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in cat2label:
                continue
            label = cat2label[name]
            # difficult = int(obj.find('difficult').text)
            difficult = False
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            # if self.min_size:
            #     assert not self.test_mode
            #     w = bbox[2] - bbox[0]
            #     h = bbox[3] - bbox[1]
            #     if w < self.min_size or h < self.min_size:
            #         ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        img_info.setdefault("ann",ann)
        img_infos.append(img_info)
    return img_infos
def parse_args():
    parser = argparse.ArgumentParser(description='underwaterdataset train dataset to mmdetction ann json')
    parser.add_argument('--dir', help='dataset root dir path')
    parser.add_argument('--out', help='output result file(pkl file)')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    assert args.out is not None and args.dir is not None
    import json
    img_infos = convert(args.dir)
    with open(args.out,"wb") as file_handler:
        pkl.dump(img_infos, file_handler)





