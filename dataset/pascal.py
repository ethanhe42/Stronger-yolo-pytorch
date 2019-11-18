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


class VOCdataset:
    def __init__(self, dataset_root, transform, subset, batchsize, trainsizes, testsize, istrain, gt_pergrid=3):
        self.dataset_root = dataset_root
        self.labels = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]
        self._transform = transform
        self._annopath = os.path.join('{}', 'Annotations', '{}.xml')
        self._imgpath = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self._ids = []
        self.testsize = testsize
        self.batch_size = batchsize
        self.multisizes = trainsizes
        self.istrain = istrain
        for year, set in subset:
            rootpath = os.path.join(dataset_root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', '{}.txt'.format(set))):
                self._ids.append((rootpath, line.strip()))
        self.strides = np.array([8, 16, 32])
        self._gt_per_grid = gt_pergrid
        self.numcls = 20

    def __len__(self):
        return len(self._ids) // self.batch_size
    def _parse_annotation(self,itemidx,random_trainsize):
        try:
            rootpath, filename = self._ids[itemidx]
        except:
            print("out of index ",itemidx)
            rootpath, filename = self._ids[10]

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
    def _load_batch(self, idx_batch, random_trainsize):
        outputshapes = random_trainsize // self.strides

        batch_image = np.zeros((self.batch_size, random_trainsize, random_trainsize, 3))
        batch_label_sbbox = np.zeros((self.batch_size, outputshapes[0], outputshapes[0],
                                      self._gt_per_grid, 6 + self.numcls))
        batch_label_mbbox = np.zeros((self.batch_size, outputshapes[1], outputshapes[1],
                                      self._gt_per_grid, 6 + self.numcls))
        batch_label_lbbox = np.zeros((self.batch_size, outputshapes[2], outputshapes[2],
                                      self._gt_per_grid, 6 + self.numcls))
        temp_batch_sbboxes = []
        temp_batch_mbboxes = []
        temp_batch_lbboxes = []
        imgpath_batch = []
        orishape_batch = []
        max_sbbox_per_img = 0
        max_mbbox_per_img = 0
        max_lbbox_per_img = 0
        for idx in range(self.batch_size):
            idxitem=idx_batch * self.batch_size + idx
            image_org,bboxes_org,labels_org,imgpath,ori_shape=self._parse_annotation(idxitem,random_trainsize)
            if random.random() < 0.5 and self.istrain:
                index_mix = random.randint(0,len(self._ids)-1)
                image_mix, bboxes_mix,label_mix,_,_ = self._parse_annotation(index_mix,random_trainsize)

                lam = np.random.beta(1.5, 1.5)
                img = lam * image_org + (1 - lam) * image_mix
                bboxes_org = np.concatenate(
                    [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=-1)
                bboxes_mix = np.concatenate(
                    [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=-1)
                bboxes = np.concatenate([bboxes_org, bboxes_mix])
                labels = np.concatenate([labels_org,label_mix])
            else:
                img = image_org
                bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=-1)
                labels=labels_org
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = \
                self.preprocess_anchorfree(bboxes, labels, outputshapes)
            batch_image[idx, :, :, :] = img
            batch_label_sbbox[idx, :, :, :, :] = label_sbbox
            batch_label_mbbox[idx, :, :, :, :] = label_mbbox
            batch_label_lbbox[idx, :, :, :, :] = label_lbbox

            zeros = np.zeros((1, 4), dtype=np.float32)
            sbboxes = sbboxes if len(sbboxes) != 0 else zeros
            mbboxes = mbboxes if len(mbboxes) != 0 else zeros
            lbboxes = lbboxes if len(lbboxes) != 0 else zeros
            temp_batch_sbboxes.append(sbboxes)
            temp_batch_mbboxes.append(mbboxes)
            temp_batch_lbboxes.append(lbboxes)
            max_sbbox_per_img = max(max_sbbox_per_img, len(sbboxes))
            max_mbbox_per_img = max(max_mbbox_per_img, len(mbboxes))
            max_lbbox_per_img = max(max_lbbox_per_img, len(lbboxes))
            imgpath_batch.append(imgpath)
            orishape_batch.append(ori_shape)

        batch_sbboxes = np.array(
            [np.concatenate([sbboxes, np.zeros((max_sbbox_per_img + 1 - len(sbboxes), 4), dtype=np.float32)], axis=0)
             for sbboxes in temp_batch_sbboxes])
        batch_mbboxes = np.array(
            [np.concatenate([mbboxes, np.zeros((max_mbbox_per_img + 1 - len(mbboxes), 4), dtype=np.float32)], axis=0)
             for mbboxes in temp_batch_mbboxes])
        batch_lbboxes = np.array(
            [np.concatenate([lbboxes, np.zeros((max_lbbox_per_img + 1 - len(lbboxes), 4), dtype=np.float32)], axis=0)
             for lbboxes in temp_batch_lbboxes])
        return torch.from_numpy(np.array(batch_image).transpose((0, 3, 1, 2)).astype(np.float32)), \
               imgpath_batch, \
               torch.from_numpy(np.array(orishape_batch).astype(np.float32)), \
               torch.from_numpy(np.array(batch_label_sbbox).astype(np.float32)), \
               torch.from_numpy(np.array(batch_label_mbbox).astype(np.float32)), \
               torch.from_numpy(np.array(batch_label_lbbox).astype(np.float32)), \
               torch.from_numpy(np.array(batch_sbboxes).astype(np.float32)), \
               torch.from_numpy(np.array(batch_mbboxes).astype(np.float32)), \
               torch.from_numpy(np.array(batch_lbboxes).astype(np.float32))

    def preprocess_anchorfree(self, bboxes, labels, outputshapes):
        '''
        :param boxes:n,x,y,x2,y2
        :param labels: n,1
        :param img_size:(h,w)
        :param class_num:
        :return:
        '''
        label = [np.zeros((outputshapes[i], outputshapes[i],
                           self._gt_per_grid, 6 + self.numcls)) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][:, :, :, -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]
        bboxes_count = [np.zeros((outputshapes[i], outputshapes[i])) for i in range(3)]
        for bbox, l in zip(bboxes, labels):
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = l
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scale = np.sqrt(np.multiply.reduce(bbox_xywh[2:]))
            bbox_mixw = bbox[4]
            # label smooth
            onehot = np.zeros(self.numcls, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.numcls, 1.0 / self.numcls)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            if bbox_scale <= 30:
                match_branch = 0
            elif (30 < bbox_scale) and (bbox_scale <= 90):
                match_branch = 1
            else:
                match_branch = 2

            xind, yind = np.floor(1.0 * bbox_xywh[:2] / self.strides[match_branch]).astype(np.int32)
            gt_count = int(bboxes_count[match_branch][yind, xind])
            if gt_count < self._gt_per_grid:
                if gt_count == 0:
                    gt_count = slice(None)
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                label[match_branch][yind, xind, gt_count, :] = 0
                label[match_branch][yind, xind, gt_count, :] = bbox_label
                bboxes_count[match_branch][yind, xind] += 1
                bboxes_coor[match_branch].append(bbox_coor)
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __getitem__(self, item):
        if self.istrain:
            trainsize = random.choice(self.multisizes)
        else:
            trainsize = self.testsize

        return self._load_batch(item, trainsize)


def get_dataset(dataset_root, batch_size, trainsizes, testsize, numworkers=4, debug=False, gt_pergrid=3):
    subset = [('2007', 'trainval'), ('2012', 'trainval')]
    datatransform = transform.YOLO3DefaultTrainTransform(mean=(0, 0, 0), std=(1, 1, 1))
    trainset = VOCdataset(dataset_root, datatransform, subset, batch_size, trainsizes=trainsizes, testsize=testsize,
                          istrain=True, gt_pergrid=gt_pergrid)
    trainset = DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=numworkers, pin_memory=True)

    subset = [('2007', 'test')]
    datatransform = transform.YOLO3DefaultValTransform(mean=(0, 0, 0), std=(1, 1, 1))
    valset = VOCdataset(dataset_root, datatransform, subset, batch_size, trainsizes=trainsizes, testsize=testsize,
                        istrain=False, gt_pergrid=gt_pergrid)

    valset = DataLoader(dataset=valset, batch_size=1, shuffle=False, num_workers=numworkers, pin_memory=True)
    return trainset, valset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, val = get_dataset('/home/gwl/datasets/VOCdevkit', 12, 416, numworkers=1)

    for batch_image, imgpath, _, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
        batch_sbboxes, batch_mbboxes, batch_lbboxes in train:
        print(batch_label_mbbox[..., 4:5].sum())
        print(batch_label_mbbox[..., 5:-1].sum())
        assert 0
