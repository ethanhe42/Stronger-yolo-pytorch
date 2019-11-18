import numpy as np
from utils.dataset_util import PascalVocXmlParser
import cv2
from dataset.augment import transform
import os
import random
import torch
from torch.utils.data import DataLoader
import os.path as osp
import dataset.augment.dataAug  as dataAug
import xml.etree.ElementTree as ET
from dataset.BaseDataset import BaseDataset

class VOCdataset(BaseDataset):
    def __init__(self,cfg,subset, istrain):
        super().__init__(cfg,subset,istrain)
        self._annopath = os.path.join('{}', 'Annotations', '{}.xml')
        self._imgpath = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self._ids = []
        for year, set in subset:
            rootpath = os.path.join(self.dataset_root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', '{}.txt'.format(set))):
                self._ids.append((rootpath, line.strip()))

    def __len__(self):
        return len(self._ids) // self.batch_size
    def _parse_annotation(self,itemidx,random_trainsize):
        rootpath, filename = self._ids[itemidx]
        annpath = self._annopath.format(rootpath, filename)
        imgpath = self._imgpath.format(rootpath, filename)
        fname, bboxes, labels = PascalVocXmlParser(annpath, self.labels).parse()
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if self.istrain:
            img, bboxes = dataAug.random_horizontal_flip(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.random_crop(np.copy(img), np.copy(bboxes))
            img, bboxes = dataAug.random_translate(np.copy(img), np.copy(bboxes))
        ori_shape=img.shape[:2]
        img, bboxes = dataAug.img_preprocess2(np.copy(img), np.copy(bboxes),
                                              (random_trainsize, random_trainsize), True)
        return img,bboxes,labels,imgpath,ori_shape

def get_dataset(cfg):
    subset = [('2007', 'trainval'), ('2012', 'trainval')]
    trainset = VOCdataset(cfg, subset,istrain=True)
    trainset = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=cfg.DATASET.numworker, pin_memory=True)

    subset = [('2007', 'test')]
    valset = VOCdataset(cfg, subset,istrain=False)
    valset = DataLoader(dataset=valset, batch_size=1, shuffle=False, num_workers=cfg.DATASET.numworker, pin_memory=True)
    return trainset, valset


if __name__ == '__main__':
    from yacscfg import _C as cfg
    import os
    import argparse
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        default='configs/strongerv1.yaml'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.EVAL.iou_thres = 0.5
    cfg.DATASET.dataset_root='/disk3/datasets/VOCdevkit'
    cfg.freeze()
    train,val=get_dataset(cfg)
    for data in train:
        print(len(train))
        assert 0
