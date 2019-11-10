import torch
from models.mobilev2 import MobileNetV2,conv_bn,sepconv_bn,conv_bias
import torch.nn as nn
from collections import OrderedDict
import pickle
class YoloV3KL(nn.Module):
    def __init__(self,numclass,gt_per_grid=3):
        super().__init__()
        self.numclass=numclass
        self.gt_per_grid=gt_per_grid
        self.mobilev2=MobileNetV2()
        load_mobilev2(self.mobilev2,'models/mobilenet_v2.pth')
        self.heads=[]
        self.headslarge=nn.Sequential(OrderedDict([
            ('conv0',conv_bn(1280,512,kernel=1,stride=1,padding=0)),
            ('conv1', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1)),
            ('conv2', conv_bn(1024, 512, kernel=1,stride=1,padding=0)),
            ('conv3', sepconv_bn(512, 1024, kernel=3, stride=1, padding=1)),
            ('conv4', conv_bn(1024, 512, kernel=1,stride=1,padding=0)),
        ]))
        self.detlarge=nn.Sequential(OrderedDict([
            ('conv5',sepconv_bn(512,1024,kernel=3, stride=1, padding=1)),
            ('conv6', conv_bias(1024, self.gt_per_grid*(self.numclass+5+4),kernel=1,stride=1,padding=0))
        ]))
        self.mergelarge=nn.Sequential(OrderedDict([
            ('conv7',conv_bn(512,256,kernel=1,stride=1,padding=0)),
            ('upsample0',nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        #-----------------------------------------------
        self.headsmid=nn.Sequential(OrderedDict([
            ('conv8',conv_bn(96+256,256,kernel=1,stride=1,padding=0)),
            ('conv9', sepconv_bn(256, 512, kernel=3, stride=1, padding=1)),
            ('conv10', conv_bn(512, 256, kernel=1,stride=1,padding=0)),
            ('conv11', sepconv_bn(256, 512, kernel=3, stride=1, padding=1)),
            ('conv12', conv_bn(512, 256, kernel=1,stride=1,padding=0)),
        ]))
        self.detmid=nn.Sequential(OrderedDict([
            ('conv13',sepconv_bn(256,512,kernel=3, stride=1, padding=1)),
            ('conv14', conv_bias(512, self.gt_per_grid*(self.numclass+5+4),kernel=1,stride=1,padding=0))
        ]))
        self.mergemid=nn.Sequential(OrderedDict([
            ('conv15',conv_bn(256,128,kernel=1,stride=1,padding=0)),
            ('upsample0',nn.UpsamplingNearest2d(scale_factor=2)),
        ]))
        #-----------------------------------------------
        self.headsmall=nn.Sequential(OrderedDict([
            ('conv16',conv_bn(32+128,128,kernel=1,stride=1,padding=0)),
            ('conv17', sepconv_bn(128, 256, kernel=3, stride=1, padding=1)),
            ('conv18', conv_bn(256, 128, kernel=1,stride=1,padding=0)),
            ('conv19', sepconv_bn(128, 256, kernel=3, stride=1, padding=1)),
            ('conv20', conv_bn(256, 128, kernel=1,stride=1,padding=0)),
        ]))
        self.detsmall=nn.Sequential(OrderedDict([
            ('conv21',sepconv_bn(128,256,kernel=3, stride=1, padding=1)),
            ('conv22', conv_bias(256, self.gt_per_grid*(self.numclass+5+4),kernel=1,stride=1,padding=0))
        ]))
    def decode(self,output,stride):
        bz=output.shape[0]
        gridsize=output.shape[-1]

        output=output.permute(0,2,3,1)
        output=output.view(bz,gridsize,gridsize,self.gt_per_grid,5+self.numclass+4)
        x1y1,x2y2,variance,conf,prob=torch.split(output,[2,2,4,1,self.numclass],dim=4)

        shiftx=torch.arange(0,gridsize,dtype=torch.float32)
        shifty=torch.arange(0,gridsize,dtype=torch.float32)
        shifty,shiftx=torch.meshgrid([shiftx,shifty])
        shiftx=shiftx.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)
        shifty=shifty.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)

        xy_grid=torch.stack([shiftx,shifty],dim=4).cuda()
        x1y1=(xy_grid+0.5-torch.exp(x1y1))*stride
        x2y2=(xy_grid+0.5+torch.exp(x2y2))*stride

        xyxy=torch.cat((x1y1,x2y2),dim=4)
        conf=torch.sigmoid(conf)
        prob=torch.sigmoid(prob)
        output=torch.cat((xyxy,variance,conf,prob),4)
        return output
    def decode_infer(self,output,stride):
        bz=output.shape[0]
        gridsize=output.shape[-1]

        output=output.permute(0,2,3,1)
        output=output.view(bz,gridsize,gridsize,self.gt_per_grid,5+self.numclass+4)
        x1y1,x2y2,variance,conf,prob=torch.split(output,[2,2,4,1,self.numclass],dim=4)

        shiftx=torch.arange(0,gridsize,dtype=torch.float32)
        shifty=torch.arange(0,gridsize,dtype=torch.float32)
        shifty,shiftx=torch.meshgrid([shiftx,shifty])
        shiftx=shiftx.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)
        shifty=shifty.unsqueeze(-1).repeat(bz,1,1,self.gt_per_grid)

        xy_grid=torch.stack([shiftx,shifty],dim=4).cuda()
        x1y1=(xy_grid+0.5-torch.exp(x1y1))*stride
        x2y2=(xy_grid+0.5+torch.exp(x2y2))*stride

        xyxy=torch.cat((x1y1,x2y2),dim=4)
        conf=torch.sigmoid(conf)
        prob=torch.sigmoid(prob)
        output=torch.cat((xyxy,variance,conf,prob),4)

        output=output.view(bz,-1,5+self.numclass+4)
        return output

    def forward(self,input):
        feat_small,feat_mid,feat_large=self.mobilev2(input)
        conv=self.headslarge(feat_large)
        outlarge=self.detlarge(conv)

        conv=self.mergelarge(conv)
        conv=self.headsmid(torch.cat((conv,feat_mid),dim=1))
        outmid=self.detmid(conv)

        conv=self.mergemid(conv)

        conv=self.headsmall(torch.cat((conv,feat_small),dim=1))
        outsmall=self.detsmall(conv)
        if self.training:
            predlarge = self.decode(outlarge, 32)
            predmid=self.decode(outmid,16)
            predsmall=self.decode(outsmall,8)
        else:
            predlarge = self.decode_infer(outlarge, 32)
            predmid = self.decode_infer(outmid, 16)
            predsmall = self.decode_infer(outsmall, 8)
            pred=torch.cat([predsmall,predmid,predlarge],dim=1)
            return pred
        return outsmall,outmid,outlarge,predsmall,predmid,predlarge
def load_mobilev2(model,ckpt):
    weights = torch.load(ckpt)
    statedict=model.state_dict()
    newstatedict=OrderedDict()
    for k,v in model.state_dict().items():
        if 'num_batches_tracked' in k:
            statedict.pop(k)
    for idx,((k,v),(k2,v2)) in enumerate(zip(statedict.items(),weights.items())):
        # print(k,v.shape,'<->',k2,v2.shape)
        newstatedict.update({k:v2})
    model.load_state_dict(newstatedict)
    print("successfully load ckpt mobilev2")
# def load_mobilev2(model,ckpt):
#     model.load_state_dict(torch.load(ckpt))
#     print("successfully load ckpt mobilev2")
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

if __name__ == '__main__':
    import pickle
    from utils.util import img_preprocess2
    import cv2
    import onnx
    import torch.onnx
    from collections import defaultdict
    from mmcv.runner import load_checkpoint
    # net=YoloV3(20)
    net=YoloV3(0)
    load_tf_weights(net,'cocoweights-half.pkl')

    assert 0
    model=net.eval()
    load_checkpoint(model,torch.load('checkpoints/coco512_prune/checkpoint-best.pth'))
    statedict=model.state_dict()
    layer2block= defaultdict(list)
    for k,v in model.state_dict().items():
        if 'num_batches_tracked' in k:
            statedict.pop(k)
    for idx,(k,v) in enumerate(statedict.items()):
        if 'mobilev2' in k:
            continue
        else:
            flag=k.split('.')[1]
            layer2block[flag].append((k,v))
    for k,v in layer2block.items():
        print(k,len(v))
    pruneratio=0.1

    # #onnx
    # input = torch.randn(1, 3, 320, 320)
    # torch.onnx.export(net, input, "coco320.onnx", verbose=True)
    # #onnxcheck
    # model=onnx.load("coco320.onnx")
    # onnx.checker.check_model(model)