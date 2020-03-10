#!/usr/bin/python

'''
生成分割训练集，生成train.txt 跟test.txt.同时检查每个xml，看是否有无object的xml。
同时生成每个image_id height  and height
'''

import os.path as osp
import os
import  argparse
import random
import xml.etree.ElementTree as ET

CLASS = ["holothurian", "echinus", "scallop", "starfish"]
test_radio = 0.15

def parse_args():
    parser = argparse.ArgumentParser(description='underwater dataset shuffle tools')
    parser.add_argument('--dir', help='dataset dir')

    #parser.add_argument('--out', help='output dir')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    data_set_dir = args.dir

    ids = os.listdir(osp.join(data_set_dir,'train','image'))
    ids = [filename.rstrip().split('.')[0] for filename in ids]

    xml_path = osp.join(data_set_dir,'train','box',"{}.xml")
    img_path = osp.join(data_set_dir,'train','image',"{}.jpg")




    final_ids = []
    illegal_list = []
    #check the  illegal ids
    for image_id in ids:
        if  checkxml(xml_path.format(image_id)):
            illegal_list.append(image_id)

    final_ids =list(set(ids) - set(illegal_list))
    test_list = random.sample(final_ids,800)
    train_list = [image_id for image_id in final_ids if image_id not in test_list]

    writelist(osp.join(data_set_dir,'train.txt'),train_list)
    writelist(osp.join(data_set_dir,'test.txt'),test_list)
    writelist(osp.join(data_set_dir,'illegal.txt'),illegal_list)




    hw_json_list = {}
    import cv2
    import json
    for image_id in final_ids:
        img_cv2 = cv2.imread(img_path.format(image_id))
        height, width, channels = img_cv2.shape
        hw_json_list.setdefault(image_id,{
            "height":height,
            "width":width
        })

    with open(osp.join(data_set_dir,"image2hw.json"),'w') as file_handler:
        json.dump(hw_json_list,file_handler)


def writelist(filename,contentlist):
    with open(filename,'w') as file_handler:
        for content in contentlist:
            file_handler.write(str(content)+"\n")






def checkxml(path):
    xml_path = osp.join(path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    illegal = True
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS:
            continue
        #label = CLASS[name]
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        # find one object ,it is not illegal
        illegal = False
        return illegal
    return illegal

if __name__ == '__main__':
    main()

