import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
  """Crop src at fixed location, and (optionally) resize it to size.

  Parameters
  ----------
  src : NDArray
      Input image
  x0 : int
      Left boundary of the cropping area
  y0 : int
      Top boundary of the cropping area
  w : int
      Width of the cropping area
  h : int
      Height of the cropping area
  size : tuple of (w, h)
      Optional, resize to new size after cropping
  interp : int, optional, default=2
      Interpolation method. See resize_short for details.

  Returns
  -------
  NDArray
      An `NDArray` containing the cropped image.
  """
  img = src[y0:y0+h,x0:x0+w,:]
  img=cv2.resize(img,(w,h))
  return img

def random_flip(src, px=0, py=0, copy=False):
  """Randomly flip image along horizontal and vertical with probabilities.

  Parameters
  ----------
  src : mxnet.nd.NDArray
      Input image with HWC format.
  px : float
      Horizontal flip probability [0, 1].
  py : float
      Vertical flip probability [0, 1].
  copy : bool
      If `True`, return a copy of input

  Returns
  -------
  mxnet.nd.NDArray
      Augmented image.
  tuple
      Tuple of (flip_x, flip_y), records of whether flips are applied.

  """
  flip_y = np.random.choice([False, True], p=[1 - py, py])
  flip_x = np.random.choice([False, True], p=[1 - px, px])
  if flip_y:
    src = np.flipud(src)
  if flip_x:
    src = np.fliplr(src)
  if copy:
    src = src.copy()
  return src, (flip_x, flip_y)



def random_color_distort(src, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                         saturation_low=0.5, saturation_high=1.5, hue_delta=18):
  """Randomly distort image color space.
  Note that input image should in original range [0, 255].

  Parameters
  ----------
  src : mxnet.nd.NDArray
      Input image as HWC format.
  brightness_delta : int
      Maximum brightness delta. Defaults to 32.
  contrast_low : float
      Lowest contrast. Defaults to 0.5.
  contrast_high : float
      Highest contrast. Defaults to 1.5.
  saturation_low : float
      Lowest saturation. Defaults to 0.5.
  saturation_high : float
      Highest saturation. Defaults to 1.5.
  hue_delta : int
      Maximum hue delta. Defaults to 18.

  Returns
  -------
  mxnet.nd.NDArray
      Distorted image in HWC format.

  """

  def brightness(src, delta, p=0.5):
    """Brightness distortion."""
    if np.random.uniform(0, 1) > p:
      delta = np.random.uniform(-delta, delta)
      src += delta
      return src
    return src

  def contrast(src, low, high, p=0.5):
    """Contrast distortion"""
    if np.random.uniform(0, 1) > p:
      alpha = np.random.uniform(low, high)
      src *= alpha
      return src
    return src

  def saturation(src, low, high, p=0.5):
    """Saturation distortion."""
    if np.random.uniform(0, 1) > p:
      alpha = np.random.uniform(low, high)
      gray = src * np.array([[[0.299, 0.587, 0.114]]])
      gray = np.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src
    return src

  def hue(src, delta, p=0.5):
    """Hue distortion"""
    if np.random.uniform(0, 1) > p:
      alpha = random.uniform(-delta, delta)
      u = np.cos(alpha * np.pi)
      w = np.sin(alpha * np.pi)
      bt = np.array([[1.0, 0.0, 0.0],
                     [0.0, u, -w],
                     [0.0, w, u]])
      tyiq = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.274, -0.321],
                       [0.211, -0.523, 0.311]])
      ityiq = np.array([[1.0, 0.956, 0.621],
                        [1.0, -0.272, -0.647],
                        [1.0, -1.107, 1.705]])
      t = np.dot(np.dot(ityiq, bt), tyiq).T
      src = np.dot(src, np.array(t))
      return src
    return src

  src = src.astype('float32')

  # brightness
  src = brightness(src, brightness_delta)

  # color jitter
  if np.random.randint(0, 2):
    src = contrast(src, contrast_low, contrast_high)
    src = saturation(src, saturation_low, saturation_high)
    src = hue(src, hue_delta)
  else:
    src = saturation(src, saturation_low, saturation_high)
    src = hue(src, hue_delta)
    src = contrast(src, contrast_low, contrast_high)
  return src

def impad_to_square(img, pad_size):
  '''Pad an image to ensure each edge to equal to pad_size.

  Args
  ---
      img: [height, width, channels]. Image to be padded
      pad_size: Int.

  Returns
  ---
      ndarray: The padded image with shape of
          [pad_size, pad_size, channels].
  '''
  shape = (pad_size, pad_size, img.shape[-1])
  pad = np.zeros(shape, dtype=img.dtype)
  pad[:img.shape[0], :img.shape[1], ...] = img
  return pad


def impad_to_multiple(img, divisor):
  '''Pad an image to ensure each edge to be multiple to some number.

  Args
  ---
      img: [height, width, channels]. Image to be padded.
      divisor: Int. Padded image edges will be multiple to divisor.

  Returns
  ---
      ndarray: The padded image.
  '''
  pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
  pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
  shape = (pad_h, pad_w, img.shape[-1])

  pad = np.zeros(shape, dtype=img.dtype)
  pad[:img.shape[0], :img.shape[1], ...] = img
  return pad


def img_resize(img, out_size):
  '''Resize image while keeping the aspect ratio.

  Args
  ---
      img: [height, width, channels]. The input image.
      out_size: Tuple of 2 integers. the image will be rescaled
          as large as possible within the scale,(w,h)

  Returns
  ---
      np.ndarray: the scaled image.
  '''
  # h, w = img.shape[:2]
  # max_long_edge = max(out_size)
  # max_short_edge = min(out_size)
  # scale_factor = min(max_long_edge / max(h, w),
  #                    max_short_edge / min(h, w))
  #
  # new_size = (int(w * float(scale_factor) + 0.5),
  #             int(h * float(scale_factor) + 0.5))

  rescaled_img = cv2.resize(
    img, out_size, interpolation=cv2.INTER_LINEAR)
  return rescaled_img


def imnormalize(img, mean, std):
  '''Normalize the image.

  Args
  ---
      img: [height, width, channel]
      mean: Tuple or np.ndarray. [3]
      std: Tuple or np.ndarray. [3]

  Returns
  ---
      np.ndarray: the normalized image.
  '''
  img=img/255.0
  img = (img - mean) / std
  return img.astype(np.float32)


def imdenormalize(norm_img, mean, std):
  '''Denormalize the image.

  Args
  ---
      norm_img: [height, width, channel]
      mean: Tuple or np.ndarray. [3]
      std: Tuple or np.ndarray. [3]

  Returns
  ---
      np.ndarray: the denormalized image.
  '''
  img = norm_img * std + mean
  return img.astype(np.float32)

def random_expand(src, max_ratio=2, keep_ratio=True):
  """Random expand original image with borders, this is identical to placing
  the original image on a larger canvas.

  Parameters
  ----------
  src : mxnet.nd.NDArray
      The original image with HWC format.
  max_ratio : int or float
      Maximum ratio of the output image on both direction(vertical and horizontal)
  fill : int or float or array-like
      The value(s) for padded borders. If `fill` is numerical type, RGB channels
      will be padded with single value. Otherwise `fill` must have same length
      as image channels, which resulted in padding with per-channel values.
  keep_ratio : bool
      If `True`, will keep output image the same aspect ratio as input.

  Returns
  -------
  mxnet.nd.NDArray
      Augmented image.
  tuple
      Tuple of (offset_x, offset_y, new_width, new_height)

  """
  if max_ratio <= 1:
    return src, (0, 0, src.shape[1], src.shape[0])

  h, w, c = src.shape
  ratio_x = random.uniform(1, max_ratio)
  if keep_ratio:
    ratio_y = ratio_x
  else:
    ratio_y = random.uniform(1, max_ratio)

  oh, ow = int(h * ratio_y), int(w * ratio_x)
  off_y = random.randint(0, oh - h)
  off_x = random.randint(0, ow - w)
  dst=np.zeros(shape=(oh,ow,c))

  dst[off_y:off_y + h, off_x:off_x + w, :] = src
  return dst, (off_x, off_y, ow, oh)

def makeImgPyramids(imgs,scales,flip=False):
  rescaled_imgs=[]
  for scale in scales:
    rescaled_img=[]
    for img in imgs:
      scaled_img=cv2.resize(img,dsize=(scale,scale))
      rescaled_img.append(scaled_img)
    rescaled_imgs.append(np.array(rescaled_img))
  if not flip:
    return rescaled_imgs
  else:
    fliped_imgs=[]
    for pyramid in rescaled_imgs:
      fliped_img=[np.fliplr(img) for img in pyramid]
      fliped_imgs.append(np.array(fliped_img))
    return rescaled_imgs+fliped_imgs

