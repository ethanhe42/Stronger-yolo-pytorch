import math
from models.backbone.baseblock import *
import torch
__all__ = ['darknet21', 'darknet53']


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.backbone_outchannels = [1024, 512, 256]
        self.inplanes = 32
        self.convbn1 = conv_bn(3, self.inplanes, kernel=3, stride=1, padding=1, activate='leaky')
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        #  downsample
        layers.append(('residual_downsample', conv_bn(self.inplanes, planes[1], kernel=3, stride=2, padding=1, activate='leaky')))
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), DarknetBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x=self.convbn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out3, out4, out5


def darknet21(pretrained=None, **kwargs):
    """Constructs a darknet-21 model.
    """
    model = DarkNet([1, 1, 2, 2, 1])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


def darknet53(pretrained=None, **kwargs):
    """Constructs a darknet-53 model.
    """
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == '__main__':
    from models.backbone.helper import *
    import os
    ckpt=torch.load('checkpoints/strongerv2_sparse/checkpoint-best.pth')
    model = darknet53(pretrained=None)
    inp = torch.ones((1, 3, 224, 224))
    out = model(inp)
    for o in out:
        print(o.shape)
