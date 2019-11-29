from utils.visualize import visualize_boxes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils.dataset_util import PascalVocXmlParser
from collections import defaultdict
import os
from .Evaluator import Evaluator


class EvaluatorVOC(Evaluator):
    def __init__(self, anchors, cateNames, rootpath, score_thres, iou_thres, use_07_metric=False):
        self.rec_pred = defaultdict(list)
        self.rec_gt = defaultdict(list)
        self.use_07_metric = use_07_metric
        self._annopath = os.path.join(rootpath, 'VOC2007', 'Annotations', '{}.xml')
        self._imgpath = os.path.join(rootpath, 'VOC2007', 'JPEGImages', '{}.jpg')
        self.reset()
        super().__init__(anchors, cateNames, rootpath, score_thres, iou_thres)

    def reset(self):
        self.coco_imgIds = set([])
        self.visual_imgs = []
        self.rec_pred = defaultdict(list)

    def append(self, imgpath, nms_boxes, nms_scores, nms_labels, visualize=True):
        if nms_boxes is not None:  # do have bboxes
            for i in range(nms_boxes.shape[0]):
                rec = {
                    "img_idx": imgpath.split('/')[-1].split('.')[0],
                    "bbox": nms_boxes[i],
                    "score": float(nms_scores[i])
                }
                self.rec_pred[nms_labels[i]].append(rec)
            if visualize and len(self.visual_imgs) < self.num_visual:
                # _, boxGT, labelGT, _ = PascalVocXmlParser(str(annpath), self.cateNames).parse()
                # boxGT=np.array(boxGT)
                # labelGT=np.array(labelGT)
                # self.append_visulize(imgpath, nms_boxes, nms_labels, nms_scores, boxGT, labelGT)
                self.append_visulize(imgpath, nms_boxes, nms_labels, nms_scores, None, None)

    def evaluate(self):
        aps = []
        for idx, cls in enumerate(self.cateNames):
            if len(self.rec_pred[idx]) > 0:
                _recs_pre = self.rec_pred[idx]
                num_recs_pre = len(_recs_pre)
                scores = np.array([rec['score'] for rec in _recs_pre])
                sorted_ind = np.argsort(-scores)
                scores = scores[sorted_ind]
                bboxs = np.array([rec['bbox'] for rec in _recs_pre])[sorted_ind]
                img_idxs = [rec['img_idx'] for rec in _recs_pre]
                img_idxs = [img_idxs[idx] for idx in sorted_ind]
                # get loggers
                num_positives = 0
                tp = np.zeros(len(img_idxs))
                fp = np.zeros(len(img_idxs))

                # build recgt according to appeard imgs
                _recs_gt = defaultdict(dict)
                for imgidx in set(img_idxs):
                    _rec = [rec for rec in self.rec_gt[imgidx] if rec['label'] == self.cateNames.index(cls)]
                    _box = np.array([rec['bbox'] for rec in _rec])
                    _dif = np.array([rec['difficult'] for rec in _rec]).astype(np.bool)
                    _detected = [False] * len(_rec)
                    num_positives += sum(~_dif)
                    _recs_gt[imgidx]['bbox'] = _box
                    _recs_gt[imgidx]['difficult'] = _dif
                    _recs_gt[imgidx]['detected'] = _detected

                # computer iou for each pred record
                for idx in range(len(img_idxs)):
                    _rec_gt = _recs_gt[img_idxs[idx]]
                    _bbGT = _rec_gt['bbox']
                    _bbPre = bboxs[idx, :]
                    ovmax = -np.inf
                    if _bbGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(_bbGT[:, 0], _bbPre[0])
                        iymin = np.maximum(_bbGT[:, 1], _bbPre[1])
                        ixmax = np.minimum(_bbGT[:, 2], _bbPre[2])
                        iymax = np.minimum(_bbGT[:, 3], _bbPre[3])

                        iw = np.maximum(ixmax - ixmin, 0.)
                        ih = np.maximum(iymax - iymin, 0.)
                        inters = iw * ih
                        uni = ((_bbPre[2] - _bbPre[0]) * (_bbPre[3] - _bbPre[1]) +
                               (_bbGT[:, 2] - _bbGT[:, 0]) *
                               (_bbGT[:, 3] - _bbGT[:, 1]) - inters)
                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)
                    # TODO add flexible threshold
                    if ovmax > self.iou_thres:
                        if not _rec_gt['difficult'][jmax]:
                            if not _rec_gt['detected'][jmax]:
                                tp[idx] = 1.
                                _rec_gt['detected'][jmax] = 1
                            else:
                                fp[idx] = 1.
                    else:
                        fp[idx] = 1.
                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(num_positives)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap = self.voc_ap(rec, prec, self.use_07_metric)
            else:
                ap = -1.
            aps.append(ap)
        return [np.mean(aps)] + aps

    def build_GT(self):
        filepath = os.path.join(self.dataset_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
        with open(filepath, 'r') as f:
            filelist = f.readlines()

        filelist = [file.strip() for file in filelist]
        for file in filelist:
            _, boxGT, labelGT, difficult = PascalVocXmlParser(self._annopath.format(file), self.cateNames).parse(
                filterdiff=False)
            for box, label, difficult in zip(boxGT, labelGT, difficult):
                self.rec_gt[file].append({
                    'label': label,
                    'bbox': box,
                    'difficult': difficult
                })

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


if __name__ == '__main__':
    dataset_root = '/disk3/datasets/voc'
    _annopath = os.path.join(dataset_root, 'VOC2007', 'Annotations', '{}.xml')
    _imgpath = os.path.join(dataset_root, 'VOC2007', 'JPEGImages', '{}.jpg')
    filepath = os.path.join(dataset_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
    with open(filepath, 'r') as f:
        filelist = f.readlines()
    cateNames = [
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
    filelist = [file.strip() for file in filelist]
    rec_gt = defaultdict(list)
    for file in filelist:
        _, boxGT, labelGT, difficult = PascalVocXmlParser(_annopath.format(file), cateNames).parse()
        for box, label, difficult in zip(boxGT, labelGT, difficult):
            rec_gt[file].append({
                'label': label,
                'bbox': box,
                'difficult': difficult
            })
    cls = "person"
    img_idxs = ['000001']
    for imgidx in set(img_idxs):
        print(rec_gt[imgidx])
        _rec = [rec for rec in rec_gt[imgidx] if rec['label'] == cls]
        print(_rec)
