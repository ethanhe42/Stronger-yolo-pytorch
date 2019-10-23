import torch
import os
from collections import OrderedDict
import numpy as np
import cv2

def ensure_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def module2weight(moduledict):
  newdict = OrderedDict()
  for k, v in moduledict.items():
    newdict.update({k.replace('module.', ''): v})
  return newdict

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    # self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 1

  def update(self, temp_sum, n=1):
    # self.val = val
    self.sum += temp_sum
    self.count += n
    # self.avg = float(self.sum)/ float(self.count)

  def get_avg(self):
    return float(self.sum) / float(self.count)

def img_preprocess2(image, bboxes, target_shape, correct_box=True, keepratio=True):
  """
  RGB转换 -> resize(resize不改变原图的高宽比) -> normalize
  并可以选择是否校正bbox
  :param image_org: 要处理的图像
  :param target_shape: 对图像处理后，期望得到的图像shape，存储格式为(h, w)
  :return: 处理之后的图像，shape为target_shape
  """
  h_target, w_target = target_shape
  h_org, w_org, _ = image.shape

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
  if not keepratio:
    ratio_w = w_target / w_org
    ratio_h = h_target / h_org
    image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR) / 255.0
    if correct_box:
      bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * ratio_w
      bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ratio_h
      return image, bboxes
    return image
  resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
  resize_w = int(resize_ratio * w_org)
  resize_h = int(resize_ratio * h_org)
  image_resized = cv2.resize(image, (resize_w, resize_h))
  image_paded = np.full((h_target, w_target, 3), 128.0)
  dw = int((w_target - resize_w) / 2)
  dh = int((h_target - resize_h) / 2)
  image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
  image = image_paded / 255.0

  if correct_box:
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
    return image, bboxes
  return image