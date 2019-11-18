from pruning.BasePruner import BasePruner
from pruning.Block import *
from models.backbone.baseblock import InvertedResidual, conv_bn, sepconv_bn,conv_bias


class l1normPruner(BasePruner):
    def __init__(self, Trainer, newmodel, pruneratio=0.1):
        super().__init__(Trainer, newmodel)
        self.pruneratio = pruneratio

    def prune(self):
        blocks = [None]
        name2layer = {}
        for midx, (name, module) in enumerate(self.model.named_modules()):
            if type(module) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn,conv_bias]:
                continue
            idx = len(blocks)
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
        blocks = blocks[1:]
        for b in blocks:
            if b.layername == 'mergelarge.conv7':
                b.inputlayer=[name2layer['headslarge.conv4']]
            if b.layername == 'headsmid.conv8':
                b.inputlayer.append(name2layer['mobilev2.features.13'])

            if b.layername == 'mergemid.conv15':
                b.inputlayer=[name2layer['headsmid.conv12']]
            if b.layername == 'headsmall.conv16':
                b.inputlayer.append(name2layer['mobilev2.features.6'])

        for b in blocks:
            if isinstance(b, CB):
                pruneweight = torch.sum(torch.abs(b.statedict[0]), dim=(1, 2, 3))
                numkeep = int(pruneweight.shape[0] * (1 - self.pruneratio))
                _ascend = torch.argsort(pruneweight)
                _descend = torch.flip(_ascend, (0,))[:numkeep]
                mask = torch.zeros_like(pruneweight).long()
                mask[_descend] = 1
                b.prunemask = torch.where(mask == 1)[0]
            if isinstance(b, InverRes):
                if b.numlayer == 2:
                    b.prunemask = torch.arange(b.outputchannel)
                if b.numlayer == 3:
                    pruneweight = torch.sum(torch.abs(b.statedict[0]), dim=(1, 2, 3))
                    numkeep = int(pruneweight.shape[0] * (1 - self.pruneratio))
                    _ascend = torch.argsort(pruneweight)
                    _descend = torch.flip(_ascend, (0,))[:numkeep]
                    mask = torch.zeros_like(pruneweight).long()
                    mask[_descend] = 1
                    b.prunemask = torch.where(mask == 1)[0]
            if isinstance(b, DCB):
                pruneweight = torch.sum(torch.abs(b.statedict[5]), dim=(1, 2, 3))
                numkeep = int(pruneweight.shape[0] * (1 - self.pruneratio))
                _ascend = torch.argsort(pruneweight)
                _descend = torch.flip(_ascend, (0,))[:numkeep]
                mask = torch.zeros_like(pruneweight).long()
                mask[_descend] = 1
                b.prunemask = torch.where(mask == 1)[0]
        blockidx = 0

        for name, m0 in self.newmodel.named_modules():
            if type(m0) not in [InvertedResidual, conv_bn, nn.Linear, sepconv_bn,conv_bias]:
                continue
            block = blocks[blockidx]
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
            if blockidx > (len(blocks) - 1): break
        print("l1 norm Pruner done")
        # print(name,block.outmask.shape)

        # for k,v in self.newmodel.state_dict().items():
        #     print(k,v.shape)
        # assert 0