# dataset settings
from torch.utils.data import DataLoader
from mmdet.datasets import  UnderWaterDataset
import  numpy as np
dataset_type = 'UnderwaterVOCDataset'
data_root = '/home/ices18/data/underwaterobjectdetection/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

#数据处理有问题,resize 尺寸有问题,同时,RandomFlip 也有问题
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='ResizeUnderWater',
        img_scale=[(2666,1600),(1333,800),(500,300)],
        multiscale_mode='value',
        keep_ratio=True),
    #dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2666,1600),(1333,800),(500,300)],
        flip=False,
        transforms=[
            dict(type='ResizeUnderWater', keep_ratio=True),

            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root + 'train/image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file=data_root + 'test.txt',
        img_prefix=data_root + 'train/image/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root = data_root,
        ann_file=data_root + 'test.txt',
        img_prefix=data_root + 'train/image/',
        pipeline=test_pipeline))
from mmdet.datasets import build_dataset

dataset  = build_dataset(data["train"])
data_loader = DataLoader(
        dataset,
        batch_size=1,
)

#可视化数据集
#问题：normal可能会影响效视觉效果，但是训练效果未知

import cv2
save_path = data_root
import os
import torch
for i in range(7):
    #img_tensor = dataset[i]['img'].data.mul_(255).add_(0.5).clamp_(0,255).permute(1,2,0).type(torch.uint8).numpy()
    filename = dataset[i]['img_meta'].data['filename']
    img_tensor = dataset[i]['img'].data.permute(1,2,0).type(torch.uint8).numpy()
    cv2_img = cv2.cvtColor(img_tensor,cv2.COLOR_RGB2BGR)
    gtbbox = dataset[i]['gt_bboxes'].data
    labels = dataset[i]['gt_labels'].data
    for index,bbox in enumerate(gtbbox):
        cv2.rectangle(cv2_img,(int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255, 0, 0), 1)
        cv2.putText(cv2_img, str(int(labels[index])), (int(bbox[0]),int(bbox[1])-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_path,os.path.basename(filename)),cv2_img)
