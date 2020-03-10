


## update log

相比原版mmdetection，修改了哪些代码：
1. backone
    * add mobilenet v2（Jan 13, 2020）
    * add shufflenet v2（ Mar 4, 2020）
2. 工程需要
    两个项目：
    1. 冰箱项目，在数据集以及config上会有refrigerator
    2. 水下目标检测比赛 underwaterdataset
    
    
    



水下目标检测比赛：

1. 根据train数据做了mmdetection的custom数据集的
一个annotation.pkl，其格式是 ：

```python
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

```
其代码是在zhoutools/underwaterdatasettoannojson.py。


2.根据训练好的结果，对a榜测试集进行测试。生成对每一张图片的标框结果图片以及提交csv文件'
图片方便更好的可视化结果。代码为：zhoutools/underwatertesttools.py
代码调用：
python3 tools/underwatertesttools.py --config configs/underwaterdataset/reppoints_moment_r50_fpn_2x_mt.py --checkpoint ./work_dirs/reppoints_moment_r50_fpn_2x_mt/latest.pth --test_dir /home/ices18/data/underwaterobjectdetection/test-A-image/ --out /home/ices18/data/underwaterobjectdetection/


可视化结果保留在 /home/ices18/data/underwaterobjectdetection/

没做的事情：
没有针对train数据集做分割出 train，test集。来验证算法结果，全部扔进去训练了；
数据预处理部分还没有动手做，全部按mmdetection的预处理来；


这几天训练过的模型跟分数：
atss_r50_fpn_1x 0.21997515
cascade_rcnn_x101_64x4d_fpn_1x：  0.38585546
cascade_rcnn_dconv_c3-c5_x101_fpn_1x：0.37622233
reppoints_moment_r50_fpn_2x_mt 0.35358598
cascade_rcnn_hrnetv2p_w32_20e.csv 0.36145455

正在跑的模型：
reppoints_moment_r101_dcn_fpn_2x_mt

主要还在找好的baseline，打算找到个不错分数的baseline（r50 backone能到0.4以上那种）
再开始改其他东西，比如数据增广，比如mutil-scale train，模型融合等等，比如数据预处理等等。

    

    
    
    