from pruning.BasePruner import BasePruner
from pruning.Block import *
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn, conv_bias,DarknetBlock


class SlimmingPruner(BasePruner):
    def __init__(self, Trainer, newmodel, cfg,savebn=''):
        super().__init__(Trainer, newmodel,cfg)
        self.pruneratio = cfg.pruneratio
        self.savebn=savebn
    def prune(self):
        super().prune()
        # gather BN weights
        bns = []
        maxbn=[]
        blacklist = [b.layername for b in self.blocks if 'residual_downsample' in b.layername]
        for b in self.blocks:
            if b.bnscale is not None and b.layername not in blacklist:
                bns.extend(b.bnscale.tolist())
                maxbn.append(b.bnscale.max().item())
        bns = torch.Tensor(bns)
        y, i = torch.sort(bns)
        if self.savebn:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.scatter(np.arange(y.shape[0])/y.shape[0],y.numpy()/y.numpy().max())
            plt.show()
            assert 0
        prunelimit=(y==min(maxbn)).nonzero().item()/len(bns)
        print("prune limit: {}".format(prunelimit))
        if self.pruneratio>prunelimit:
            raise AssertionError('prune ratio bigger than limit')
        thre_index = int(bns.shape[0] * self.pruneratio)
        thre = y[thre_index]
        thre = thre.cuda()
        pruned_bn = 0
        for b in self.blocks:
            if isinstance(b, CB):
                ## for darknet pruing, residual_downsample's output must be kept
                if 'residual_downsample' in b.layername:
                    mask = torch.ones_like(b.bnscale)
                    b.prunemask = torch.arange(b.bnscale.shape[0])
                else:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, InverRes):
                if b.numlayer == 3:
                    mask = b.bnscale.gt(thre)
                    pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                    b.prunemask = torch.where(mask == 1)[0]
                    print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b, DCB):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
            if isinstance(b,DarkBlock):
                mask = b.bnscale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask)
                b.prunemask = torch.where(mask == 1)[0]
                print("{}:{}/{} pruned".format(b.layername, mask.shape[0] - torch.sum(mask), mask.shape[0]))
        blockidx = 0
        for name, m0 in self.newmodel.named_modules():
            if type(m0) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn,conv_bias,DarknetBlock]:
                continue
            block = self.blocks[blockidx]
            curstatedict = block.statedict
            if (len(block.inputlayer) == 1):
                if block.inputlayer[0] is None:
                    inputmask = torch.arange(block.inputchannel)
                else:
                    inputmask = block.inputlayer[0].outmask
            elif (len(block.inputlayer) == 2):
                first = block.inputlayer[0].outmask
                second = block.inputlayer[1].outmask
                second+=block.inputlayer[0].outputchannel
                second=second.to(first.device)
                inputmask=torch.cat((first,second),0)
            else:
                raise AttributeError
            if isinstance(block,DarkBlock):
                assert len(curstatedict)==(1+4+1+4)
                block.clone2module(m0,inputmask)
            if isinstance(block, CB):
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4)
                block.clone2module(m0, inputmask)
            if isinstance(block, DCB):
                # conv(1weight)->bn(4weight)->relu
                assert len(curstatedict) == (1 + 4 + 1 + 4)
                block.clone2module(m0, inputmask)
            if isinstance(block, InverRes):
                # dw->project or expand->dw->project
                assert len(curstatedict) in (10, 15)
                block.clone2module(m0, inputmask)
            if isinstance(block, FC):
                block.clone2module(m0)
            if isinstance(block, Conv):
                block.clone2module(m0,inputmask)

            blockidx += 1
            if blockidx > (len(self.blocks) - 1): break
        print("Slimming Pruner done")