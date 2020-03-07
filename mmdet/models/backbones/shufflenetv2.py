import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint

from ..registry import BACKBONES

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


def ScaleRatioToStageOutChannels(ratio):
    if ratio==0.5:
        return [24, 48, 96, 192]
    elif ratio == 1:
        return [24, 116, 232, 464]
    elif ratio == 1.5:
        return [24, 176, 352, 704]
    elif ratio == 2.0:
        return [24, 244, 488, 976]

@BACKBONES.register_module
class ShuffleNetV2(nn.Module):


    '''
    shuffleNet v2 remove the final conv layer and turn to 4 stage features out
    scaleratio: the out channles scale: chosen in [0.5,1,1.5,2]
    the group channels out will be with 0.5  ->[24, 48, 96, 192]
                                        1.0 -> [24, 116, 232, 464]
                                        1.5 -> [24, 176, 352, 704]
                                        2   -> [24, 244, 488, 976]
    '''
    def __init__(self, scaleratio=1, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        stages_repeats = [4,8,4]
        if scaleratio not in [0.5,1,1.5,2]:
            raise ValueError("scaleratio should be chosen in [0.5,1,1.5,2]")

        self._stage_out_channels = ScaleRatioToStageOutChannels(scaleratio)

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                self.stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels




    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        outs.append(x)
        for i,stage_name in enumerate(self.stage_names):
            stage = getattr(self,stage_name)
            x = stage(x)
            outs.append(x)
        return tuple(outs)


