from dataset.pycocotools.coco import COCO
from dataset.pycocotools.cocoeval import COCOeval
import os
from .Evaluator import Evaluator
import numpy as np


class EvaluatorCOCO(Evaluator):
    def __init__(self, anchors, cateNames, rootpath, score_thres, iou_thres):
        self.coco_imgIds = set([])
        self.coco_results = []
        self.idx2cat = {
            "0": 1,
            "1": 2,
            "2": 3,
            "3": 4,
            "4": 5,
            "5": 6,
            "6": 7,
            "7": 8,
            "8": 9,
            "9": 10,
            "10": 11,
            "11": 13,
            "12": 14,
            "13": 15,
            "14": 16,
            "15": 17,
            "16": 18,
            "17": 19,
            "18": 20,
            "19": 21,
            "20": 22,
            "21": 23,
            "22": 24,
            "23": 25,
            "24": 27,
            "25": 28,
            "26": 31,
            "27": 32,
            "28": 33,
            "29": 34,
            "30": 35,
            "31": 36,
            "32": 37,
            "33": 38,
            "34": 39,
            "35": 40,
            "36": 41,
            "37": 42,
            "38": 43,
            "39": 44,
            "40": 46,
            "41": 47,
            "42": 48,
            "43": 49,
            "44": 50,
            "45": 51,
            "46": 52,
            "47": 53,
            "48": 54,
            "49": 55,
            "50": 56,
            "51": 57,
            "52": 58,
            "53": 59,
            "54": 60,
            "55": 61,
            "56": 62,
            "57": 63,
            "58": 64,
            "59": 65,
            "60": 67,
            "61": 70,
            "62": 72,
            "63": 73,
            "64": 74,
            "65": 75,
            "66": 76,
            "67": 77,
            "68": 78,
            "69": 79,
            "70": 80,
            "71": 81,
            "72": 82,
            "73": 84,
            "74": 85,
            "75": 86,
            "76": 87,
            "77": 88,
            "78": 89,
            "79": 90
        }
        self.cat2idx = {int(v): int(k) for k, v in self.idx2cat.items()}
        self.reset()
        super().__init__(anchors, cateNames, rootpath, score_thres, iou_thres)

    def reset(self):
        self.coco_imgIds = set([])
        self.coco_results = []
        self.visual_imgs = []

    def build_GT(self):
        self.cocoGt = COCO(os.path.join(self.dataset_root, 'annotations/instances_val2017.json'))

    def append(self, imgpath, nms_boxes, nms_scores, nms_labels, visualize=True):
        imgid = int(imgpath[-16:-4])
        if nms_boxes is not None:  # do have bboxes
            for i in range(nms_boxes.shape[0]):
                self.coco_imgIds.add(imgid)
                self.coco_results.append({
                    "image_id": imgid,
                    "category_id": self.idx2cat[str(nms_labels[i])],
                    "bbox": [nms_boxes[i][0], nms_boxes[i][1], nms_boxes[i][2] - nms_boxes[i][0],
                             nms_boxes[i][3] - nms_boxes[i][1]],
                    "score": float(nms_scores[i])
                })
            if len(self.visual_imgs) < self.num_visual:
                annIDs = self.cocoGt.getAnnIds(imgIds=[imgid])
                boxGT = []
                labelGT = []
                for id in annIDs:
                    ann = self.cocoGt.anns[id]
                    x, y, w, h = ann['bbox']
                    boxGT.append([x, y, x + w, y + h])
                    labelGT.append(self.cat2idx[ann['category_id']])
                boxGT = np.array(boxGT)
                labelGT = np.array(labelGT)
                self.append_visulize(imgpath, nms_boxes, nms_labels, nms_scores, boxGT, labelGT)

    def evaluate(self):
        try:
            cocoDt = self.cocoGt.loadRes(self.coco_results)
        except:
            print("no boxes detected, coco eval aborted")
            return [0.0]*12
        cocoEval = COCOeval(self.cocoGt, cocoDt, "bbox")
        cocoEval.params.imgIds = list(self.coco_imgIds)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return cocoEval.stats
