import os
import torch
from trainers.base_trainer import BaseTrainer
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn,conv_bias,DarknetBlock
from pruning.Block import *

class BasePruner:
    def __init__(self,trainer:BaseTrainer,newmodel,cfg):
        self.model=trainer.model
        self.newmodel=newmodel
        self.trainer=trainer
        self.blocks=[]
        self.pruneratio = 0.1
        self.args=cfg
    def prune(self):
        blocks = [None]
        name2layer = {}
        for midx, (name, module) in enumerate(self.model.named_modules()):
            if type(module) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn, conv_bias,DarknetBlock]:
                continue
            idx = len(blocks)
            if isinstance(module,DarknetBlock):
                blocks.append(DarkBlock(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, InvertedResidual):
                blocks.append(InverRes(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, conv_bn):
                blocks.append(CB(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, nn.Linear):
                blocks.append(FC(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, sepconv_bn):
                blocks.append(DCB(name, idx, [blocks[-1]], list(module.state_dict().values())))
            if isinstance(module, conv_bias):
                blocks.append(Conv(name, idx, [blocks[-1]], list(module.state_dict().values())))
            name2layer[name] = blocks[-1]
        self.blocks = blocks[1:]
        for b in self.blocks:
            if b.layername == 'mergelarge.conv7':
                b.inputlayer=[name2layer['headslarge.conv4']]
            if b.layername == 'headsmid.conv8':
                b.inputlayer.append(name2layer[self.args.bbOutName[1]])
            if b.layername == 'mergemid.conv15':
                b.inputlayer=[name2layer['headsmid.conv12']]
            if b.layername == 'headsmall.conv16':
                b.inputlayer.append(name2layer[self.args.bbOutName[0]])
    def test(self,newmodel=False,validiter=20):
        if newmodel:
            self.trainer.model=self.newmodel
        results,_=self.trainer._valid_epoch(validiter=validiter)
        self.trainer.TESTevaluator.reset()
        return results[0]
    def finetune(self,epoch=10):
        self.trainer.model=self.newmodel
        # self.best_mAP=self.trainer._valid_epoch(validiter=10)[0][0]
        self.best_mAP=0
        for epoch in range(0, self.trainer.args.OPTIM.total_epoch):
            self.trainer.global_epoch += 1
            self.trainer._train_epoch()
            self.trainer.lr_scheduler.step(epoch)
            lr_current = self.trainer.optimizer.param_groups[0]['lr']
            print("epoch:{} lr:{}".format(epoch,lr_current))
            results, imgs = self.trainer._valid_epoch()
            self.trainer._reset_loggers()
            if results[0] > self.best_mAP:
                self.best_mAP = results[0]
                self.trainer._save_ckpt(name='best-ft{}'.format(self.pruneratio), metric=self.best_mAP)
        return self.best_mAP
