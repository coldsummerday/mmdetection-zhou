#!/usr/bin/python

"""
该脚本是对训练结果测试集的可视化图片生成并生成提交文件：

可视化结果为对图片进行画框保存：
文件中每行结果为：
classname,imageid,confidence,xmin,ymin,xmax,ymax
"""
import argparse
import os
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("test_dir",help="testimage dir")
    parser.add_argument('--out', help='output dir')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    save_path = os.path.join(args.out,os.path.basename(args.config).split(".")[0])
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    imagefilenames = os.listdir(args.test_dir)
    imagefilenames = [os.path.join(args.test_dir) for image in imagefilenames]
    total_lines = []
    for image in imagefilenames:
        img_cv2 = cv2.imread(image)
        base_name = os.path.basename(image)
        result = inference_detector(model, img_cv2)
        bboxs = result2objects(result,model.CLASSES,score_threshold=0.5)
        img_cv2 = draw_bbox(img_cv2,bboxs)
        total_lines.extend(bboxs2lines(base_name.split(".")[0],bboxs))
        cv2.imwrite(os.path.join(save_path,base_name),img_cv2)

    with open(os.path.join(save_path,"image_submission.csv"),'w') as file_handler:
        for line in total_lines:
            file_handler.write(line)



def bboxs2lines(imageid,bboxs):
    result = []
    for box in bboxs:
        result.append(
            "{},{},{},{},{},{},{}\n".format(box["label"],str(imageid)+".xml",box["confidence"],
                                     box["x1"],box["y1"],box["x2"],box["y2"])
        )
    return result

def draw_bbox(img_cv2,bboxs):
    for box_object in bboxs:
        left_top = (box_object['x1'], box_object['y1'])
        right_bottom = (box_object['x2'], box_object['y2'])
        cv2.rectangle(
            img_cv2, left_top, right_bottom, (255, 0, 0), 2)
        label_text = "{}:{.2f}".format(box_object['label'],box_object["confidence"])

        cv2.putText(img_cv2, label_text, (box_object['x1'],box_object['y1'] - 2),
                    cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),3)
    return img_cv2

def result2objects(result,classes,score_threshold=0.5):
    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    # 过滤掉分数过低的目标框
    inds = bboxes[:, -1] > score_threshold

    bboxes = bboxes[inds, :]
    labels = labels[inds]
    box_objects = []
    for bbox, label in zip(bboxes, labels):
        object_dict = {"label": classes[label],
                       "x1": int(bbox[0]),
                       "y1": int(bbox[1]),
                       "x2": int(bbox[2]),
                       "y2": int(bbox[3]),
                       "confidence": float(bbox[-1])}
        box_objects.append(object_dict)
    return box_objects
"""
        image_cv2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        ori_height, ori_width, _ = image_cv2.shape
        result = inference_detector(model, image_cv2)

        
        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        # 过滤掉分数过低的目标框
        inds = bboxes[:, -1] > score_threshold

        bboxes = bboxes[inds, :]
        labels = labels[inds]
        box_objects = []
        for bbox, label in zip(bboxes, labels):
            object_dict = {"label": model.CLASSES[label],
                           "x1": int(bbox[0]),
                           "y1": int(bbox[1]),
                           "x2": int(bbox[2]),
                           "y2": int(bbox[3]),
                           "confidence": float(bbox[-1])}
"""