from yacs.config import CfgNode as CN
_C = CN()
_C.debug= False
_C.do_test=False
_C.finetune=False
_C.devices= (0,)

_C.MODEL=CN()
_C.MODEL.LABEL=[]
_C.MODEL.modeltype = 'YoloV3'
_C.MODEL.backbone = 'mobilenetv2'
_C.MODEL.multiwidth=1.0
_C.MODEL.backbone_pretrained=''
_C.MODEL.numcls=20
_C.MODEL.gt_per_grid=3
_C.MODEL.clsfocal = False
_C.MODEL.seprelu = True
_C.MODEL.boxloss = 'iou'
_C.MODEL.l1scale = 1.0
_C.MODEL.ASFF=False
_C.EVAL=CN()
#iou thres for VOC,default is map50
_C.EVAL.iou_thres=0.5
_C.EVAL.nms_iou=0.45
_C.EVAL.softnms=False
_C.EVAL.varvote=False
_C.EVAL.vvsigma=0.05

_C.EVAL.score_thres=0.1
_C.EVAL.soft=False
_C.EVAL.softsigma=False

_C.EXPER=CN()
_C.EXPER.experiment_name=''
_C.EXPER.train_sizes=[480,512,544]
_C.EXPER.test_size=544
_C.EXPER.resume=''

_C.OPTIM=CN()
_C.OPTIM.batch_size=12
_C.OPTIM.lr_initial=2e-4
_C.OPTIM.total_epoch=60
_C.OPTIM.milestones=[30,45]

_C.DATASET=CN()
_C.DATASET.dataset= 'VOC'
_C.DATASET.dataset_root='/home/gwl/datasets/VOCdevkit'
_C.DATASET.numworker=4

_C.LOG=CN()
_C.LOG.log_iter=200

_C.Prune=CN()
_C.Prune.sparse=False
_C.Prune.sr=0.01
_C.Prune.pruneratio=0.0
_C.Prune.bbOutName=('backbone.layer3.residual_7','backbone.layer4.residual_7','backbone.layer5.residual_3')
_C.Prune.do_test=False



