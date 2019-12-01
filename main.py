from models import *
from trainers import *
import json
from yacscfg import _C as cfg
import os
from torch import optim
import argparse
import torch
def main(args):
    gpus=[str(g) for g in args.devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    net = eval(cfg.MODEL.modeltype)(cfg=args.MODEL).cuda()
    optimizer = optim.Adam(net.parameters(),lr=args.OPTIM.lr_initial)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.OPTIM.milestones, gamma=0.1)
    _Trainer = eval('Trainer_{}'.format(args.DATASET.dataset))(args=args,
                       model=net,
                       optimizer=optimizer,
                       lrscheduler=scheduler
                       )
    if args.do_test:
      _Trainer._valid_epoch(validiter=-1)
    else:
      _Trainer.train()

  #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DEMO configuration")
    parser.add_argument(
        "--config-file",
        default = 'configs/strongerv3_all.yaml'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print(cfg)
    main(args=cfg)
