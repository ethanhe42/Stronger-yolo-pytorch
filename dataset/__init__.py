from .coco import get_dataset as get_COCO
from .pascal import get_dataset as get_VOC
from dataset.augment.bbox import bbox_flip
from dataset.augment.image import makeImgPyramids
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

def get_imgdir(dataset_root, batch_size, net_size):
  from torchvision.transforms import transforms
  from PIL import Image
  class dataset:
    def __init__(self, root, transform):
      self.imglist = os.listdir(root)
      self.root = root
      self.transform = transform

    def __len__(self):
      return len(self.imglist)

    def __getitem__(self, item):
      path=os.path.join(self.root,self.imglist[item])
      img=Image.open(path)
      ori_shape=np.array(img.size)
      img=self.transform(img)
      return path,img,torch.from_numpy(ori_shape.astype(np.float32))
  transform=transforms.Compose([
    transforms.Resize((net_size,net_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,0,0),std=(1,1,1))
  ])

  dataloader=DataLoader(dataset=dataset(dataset_root,transform),shuffle=False,batch_size=batch_size)
  return dataloader