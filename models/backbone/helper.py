import torch
from collections import OrderedDict
import pickle
def load_mobilev2(model,ckpt):
    weights = torch.load(ckpt)
    statedict=model.state_dict()
    newstatedict=OrderedDict()
    for k,v in model.state_dict().items():
        if 'num_batches_tracked' in k:
            statedict.pop(k)
    for idx,((k,v),(k2,v2)) in enumerate(zip(statedict.items(),weights.items())):
        newstatedict.update({k:v2})
    model.load_state_dict(newstatedict)
    print("successfully load ckpt mobilev2")
def load_tf_weights(model,ckpt):
    with open(ckpt, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')
    statedict=model.state_dict()
    for k,v in model.state_dict().items():
        if 'num_batches_tracked' in k:
            statedict.pop(k)
    newstatedict=OrderedDict()
    for idx,((k,v),(k2,v2)) in enumerate(zip(statedict.items(),weights.items())):
        if v.ndim>1:
            if 'depthwise' in k2:
                #hwio->iohw
                newstatedict.update({k:torch.from_numpy(v2.transpose(2,3,0,1))})
            else:
                #hwoi->iohw
                newstatedict.update({k:torch.from_numpy(v2.transpose(3,2,0,1))})
        else:
            newstatedict.update({k:torch.from_numpy(v2)})
    model.load_state_dict(newstatedict)
    print("successfully load ckpt")
def load_darknet_weights(model, weights_path):
  import numpy as np
  # Open the weights file
  fp = open(weights_path, "rb")
  header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values
  # Needed to write header when saving weights
  weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
  fp.close()

  ptr = 0
  all_dict = model.state_dict()
  last_bn_weight = None
  last_conv = None
  for i, (k, v) in enumerate(all_dict.items()):
    if 'bn' in k:
      if 'weight' in k:
        last_bn_weight = v
      elif 'bias' in k:
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
        # weight
        v = last_bn_weight
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
        last_bn_weight = None
      elif 'running_mean' in k:
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
      elif 'running_var' in k:
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
        # conv
        v = last_conv
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
        last_conv = None
      elif 'num_batches_tracked' in k:
        continue
      else:
        raise Exception("Error for bn")
    elif 'conv' in k:
      if 'weight' in k:
        last_conv = v
      else:
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
        # conv
        v = last_conv
        num_b = v.numel()
        vv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(v)
        v.copy_(vv)
        ptr += num_b
        last_conv = None
  print("Total ptr = ", ptr)
  print("real size = ", weights.shape)
