# /usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector
import argparse
import base64
import io
import json
import logging as log
import os

from io import BytesIO as Bytes2Data

import werkzeug
from flask import Flask
from flask_restful import Api, Resource, reqparse


# 用tornado 部署 flask
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


log.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=log.DEBUG)


config_file = 'configs/refrigerator/fcos_r50_caffe_rpn_gn_1x_4gpu.py'
checkpoint_file = 'work_dirs/fcos_epoch_14.pth'
score_threshold = 0.3
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


class ImageDetect(Resource):
    image_parser = reqparse.RequestParser()
    image_parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files', required=True, action='append',
                              help="Can't find image parameter")

    def post(self):
        args = self.image_parser.parse_args()

        image_file = args.image[0]

        ##直接从二进制流中读取图片文件,避免二次IO
        image_name = image_file.filename
        image_data = image_file.read()
        image_bytes = bytearray(image_data)
        image_array = np.asarray(image_bytes)
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
            box_objects.append(object_dict)
        final_result = {
            "filename": image_file.filename,
            "width": ori_height,
            "height": ori_width,
            "detectresult": box_objects
        }
        return {
            "filename": image_file.filename,
            "width": ori_height,
            "height": ori_width,
            "detectresult": box_objects
        }


if __name__ == '__main__':
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(ImageDetect, '/detect')
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000, "0.0.0.0")
    IOLoop.instance().start()
