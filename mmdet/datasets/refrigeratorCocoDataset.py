
from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class RefrigeratorCocoDataset(CocoDataset):
    CLASSES = ("fenda", "yingyangkuaixian", "jiaduobao", "maidong","TYCL", "BSS", "TYYC", "LLDS", "KSFH", "MZY")
