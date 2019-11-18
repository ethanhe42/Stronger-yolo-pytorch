import os
import os.path as osp
import cv2
import numpy as np
from dataset.pycocotools.coco import COCO
from dataset.augment import transform
import random
import torch.utils.data as data
import torch
import dataset.augment.dataAug  as dataAug
from dataset.BaseDataset import BaseDataset

class COCOdataset(BaseDataset):
    def __init__(self, cfg,subset,istrain):
        super().__init__(cfg,subset,istrain)
        self.image_dir = "{}/images/{}2017".format(self.dataset_root, subset)
        self.coco = COCO("{}/annotations/instances_{}2017.json".format(self.dataset_root, subset))
        # get the mapping from original category ids to labels
        self.cat_ids = self.coco.getCatIds()
        self.numcls = len(self.cat_ids)
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids, self.img_infos = self._filter_imgs()

    def _filter_imgs(self, min_size=32):
        # Filter images without ground truths.
        all_img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        # Filter images too small.
        img_ids = []
        img_infos = []
        for i in all_img_ids:
            info = self.coco.loadImgs(i)[0]
            ann_ids = self.coco.getAnnIds(imgIds=i)
            ann_info = self.coco.loadAnns(ann_ids)
            ann = self._parse_ann_info(ann_info)
            if min(info['width'], info['height']) >= min_size and ann['labels'].shape[0] != 0:
                img_ids.append(i)
                img_infos.append(info)
        return img_ids, img_infos

    def _load_ann_info(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann_info = self.coco.loadAnns(ann_ids)
        return ann_info

    def _parse_ann_info(self, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        return ann

    def __len__(self):
        return len(self.img_infos) // self.batch_size

    def _parse_annotation(self,itemidx,random_trainsize):
        img_info = self.img_infos[itemidx]
        ann_info = self._load_ann_info(itemidx)
        ann = self._parse_ann_info(ann_info)
        bboxes = ann['bboxes']  # [x1,y1,x2,y2]
        labels = ann['labels']
        # load the image.
        imgpath=osp.join(self.image_dir, img_info['file_name'])
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
    valset = COCOdataset(cfg, subset='val',istrain=False)
    valset = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=True, num_workers=cfg.DATASET.numworker, pin_memory=True)
    if cfg.debug:
        return valset, valset
    trainset = COCOdataset(cfg, subset='train',istrain=True)
    trainset = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=cfg.DATASET.numworker, pin_memory=True)
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
    cfg.DATASET.dataset_root='/disk3/datasets/coco'
    cfg.debug=True
    cfg.freeze()
    train,val=get_dataset(cfg)
    for data in val:
        print(len(train))
        assert 0