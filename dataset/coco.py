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


class COCOdataset(data.Dataset):
    def __init__(self, dataset_root, transform, subset, batchsize,trainsizes, testsize, istrain,gt_pergrid=3):
        self.dataset_root = dataset_root
        self.image_dir = "{}/images/{}2017".format(dataset_root, subset)
        self.coco = COCO("{}/annotations/instances_{}2017.json".format(dataset_root, subset))
        self.istrain = istrain
        self.testsize = testsize
        self.batch_size = batchsize
        # get the mapping from original category ids to labels
        self.cat_ids = self.coco.getCatIds()
        self.numcls = len(self.cat_ids)
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids, self.img_infos = self._filter_imgs()
        self._transform = transform
        self.multisizes = trainsizes
        self.strides = np.array([8, 16, 32])
        self._gt_per_grid = gt_pergrid

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

    def __getitem__(self, index):
        if self.istrain:
            trainsize = random.choice(self.multisizes)
        else:
            trainsize = self.testsize

        return self._load_batch(index, trainsize)

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
            img_info = self.img_infos[idx_batch * self.batch_size + idx]
            ann_info = self._load_ann_info(idx_batch * self.batch_size + idx)
            # load the image.
            img = cv2.imread(osp.join(self.image_dir, img_info['file_name']), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_shape = img.shape[:2]  # yx-->xy
            # Load the annotation.
            ann = self._parse_ann_info(ann_info)
            bboxes = ann['bboxes']  # [x1,y1,x2,y2]
            labels = ann['labels']
            img, bboxes = self._transform(random_trainsize, random_trainsize, img, bboxes)
            # data augmentation in original-strongeryolo
            # if self.istrain:
            #     img, bboxes = dataAug.random_horizontal_flip(np.copy(img), np.copy(bboxes))
            #     img, bboxes = dataAug.random_crop(np.copy(img), np.copy(bboxes))
            #     img, bboxes = dataAug.random_translate(np.copy(img), np.copy(bboxes))
            # img, bboxes = dataAug.img_preprocess2(np.copy(img), np.copy(bboxes),
            #                                       (random_trainsize, random_trainsize), True)

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
            imgpath_batch.append(osp.join(self.image_dir, img_info['file_name']))
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

        for bbox in bboxes:
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = labels
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scale = np.sqrt(np.multiply.reduce(bbox_xywh[2:]))

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
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot,[1]], axis=-1)
                label[match_branch][yind, xind, gt_count, :] = 0
                label[match_branch][yind, xind, gt_count, :] = bbox_label
                bboxes_count[match_branch][yind, xind] += 1
                bboxes_coor[match_branch].append(bbox_coor)
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


def get_dataset(dataset_root, batch_size, trainsizes,testsize,numworkers=4,debug=False,gt_pergrid=3):
    datatransform = transform.YOLO3DefaultValTransform(mean=(0, 0, 0), std=(1, 1, 1))
    valset = COCOdataset(dataset_root, datatransform, subset='val', batchsize=batch_size,trainsizes=trainsizes, testsize=testsize,
                         istrain=False,gt_pergrid=gt_pergrid)
    valset = torch.utils.data.DataLoader(dataset=valset, batch_size=1, shuffle=False, num_workers=numworkers, pin_memory=True)
    if debug:
        return valset, valset
    trainset = COCOdataset(dataset_root, datatransform, subset='train', batchsize=batch_size,trainsizes=trainsizes, testsize=testsize,
                           istrain=True,gt_pergrid=gt_pergrid)
    trainset = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=numworkers, pin_memory=True)
    return trainset, valset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train, val = get_dataset('/datasets/coco', 16, 416,numworkers=1,debug=True)
    for batch_image,imgpath,_, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes in train:
        print (batch_label_sbbox.shape)
        assert 0
