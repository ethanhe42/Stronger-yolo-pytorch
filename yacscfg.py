from yacs.config import CfgNode as CN
_C = CN()
_C.debug= False
_C.do_test=False
_C.finetune=False
_C.devices= (0,)

_C.MODEL=CN()
_C.MODEL.LABEL=[]
_C.MODEL.numcls=20
_C.MODEL.gt_per_grid=3

_C.EXPER=CN()
_C.EXPER.experiment_name=''
_C.EXPER.train_sizes=[480,512,544]
_C.EXPER.test_size=512
_C.EXPER.resume=''

_C.OPTIM=CN()
_C.OPTIM.batch_size=12
_C.OPTIM.lr_initial=2e-4
_C.OPTIM.total_epoch=60
_C.OPTIM.milestones=[30,45]

_C.DATASET=CN()
_C.DATASET.dataset= 'VOC'
_C.DATASET.dataset_root='/home/gwl/datasets/VOCdevkit'

_C.LOG=CN()
_C.LOG.log_iter=200

_C.Prune=CN()
_C.Prune.sparse=False
_C.Prune.sr=0.01
_C.Prune.pruneratio=0.0




