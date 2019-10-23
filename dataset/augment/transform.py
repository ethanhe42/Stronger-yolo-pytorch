import numpy as np

import dataset.augment.image as timage
import dataset.augment.bbox as tbbox


class YOLO3DefaultValTransform(object):
    """Default YOLO validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std

    def __call__(self, width, height, img, bbox):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = img.shape
        img = timage.img_resize(img, out_size=(width, height))
        bbox = tbbox.bbox_resize(bbox, (w, h), (width, height))
        img = timage.imnormalize(img, self._mean, self._std)
        return img, bbox.astype(img.dtype)


class YOLO3DefaultTrainTransform(object):
    def __init__(self, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):

        self._mean = mean
        self._std = std

    def __call__(self, width, height, img, bbox):
        """
        :param image:np.array HWC
        :param bbox:np.array box N,4 x1y1x2y2
        :return:
        """
        img = timage.random_color_distort(img)
        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0:
            img, expand = timage.random_expand(img)
            bbox = tbbox.translate(bbox, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, bbox

        # random cropping
        h, w, _ = img.shape
        bbox, crop = tbbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = timage.fixed_crop(img, x0, y0, w, h)

        # resize
        img = timage.img_resize(img, out_size=(width, height))
        bbox = tbbox.bbox_resize(bbox, (w, h), (width, height))

        # flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=1)
        bbox = tbbox.bbox_flip(bbox, (w, h), flip_x=flips[0])

        # normalize
        img = timage.imnormalize(img, self._mean, self._std)

        return img, bbox

    def denormalize(self, img):
        return timage.imdenormalize(img, self._mean, self._std)


def preprocess(boxes, labels, input_shape, class_num, anchors):
    '''
    :param boxes:n,x,y,x2,y2
    :param labels: n,1
    :param img_size:(h,w)
    :param class_num:
    :param anchors:(9,2)
    :return:
    '''
    input_shape = np.array(input_shape)
    # find match anchor for each box,leveraging numpy broadcasting tricks
    boxes_center = (boxes[..., 2:4] + boxes[..., 0:2]) // 2
    boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
    boxes_wh = np.expand_dims(boxes_wh, 1)
    min_wh = np.maximum(-boxes_wh / 2, -anchors / 2)
    max_wh = np.minimum(boxes_wh / 2, anchors / 2)
    intersect_wh = max_wh - min_wh
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    anchors_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchors_area - intersect_area)
    best_ious = np.argmax(iou, axis=1)
    # normalize boxes according to inputsize(416)
    boxes[..., 0:2] = boxes_center / input_shape[::-1]
    boxes[..., 2:4] = np.squeeze(boxes_wh, 1) / input_shape[::-1]
    # get dummy gt with zeros
    y_true_52 = np.zeros((input_shape[1] // 8, input_shape[0] // 8, 3, 5 + class_num), np.float32)
    y_true_26 = np.zeros((input_shape[1] // 16, input_shape[0] // 16, 3, 5 + class_num), np.float32)
    y_true_13 = np.zeros((input_shape[1] // 32, input_shape[0] // 32, 3, 5 + class_num), np.float32)
    y_true_list = [y_true_52, y_true_26, y_true_13]
    grid_shapes = [input_shape // 8, input_shape // 16, input_shape // 32]
    for idx, match_id in enumerate(best_ious):
        group_idx = match_id // 3
        sub_idx = match_id % 3
        idx_x = np.floor(boxes[idx, 0] * grid_shapes[group_idx]).astype('int32')
        idx_y = np.floor(boxes[idx, 1] * grid_shapes[group_idx]).astype('int32')

        y_true_list[group_idx][idx_y, idx_x, sub_idx, :2] = boxes[idx, 0:2]
        y_true_list[group_idx][idx_y, idx_x, sub_idx, 2:4] = boxes[idx, 2:4]
        y_true_list[group_idx][idx_y, idx_x, sub_idx, 4] = 1.
        y_true_list[group_idx][idx_y, idx_x, sub_idx, 5 + labels[idx]] = 1.
    return y_true_list
