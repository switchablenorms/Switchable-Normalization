import torch.nn as nn
import torch.nn.functional as F
import math
# from . import NormFunc
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50','resnet100','resnet101',
           'resnet152']

NormFunc = nn.BatchNorm2d

class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, bottle_neck=False, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        if bottle_neck:
          self.norm1 = NormFunc(inplanes)
          self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
          self.norm2 = NormFunc(planes)
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
          self.norm3 = NormFunc(planes)
          self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
          self.norm4 = NormFunc(planes * 4)
          self.prelu = nn.PReLU()
        else:
          self.norm1 = NormFunc(inplanes)
          self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
          self.norm2 = NormFunc(planes )
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
          self.norm3 = NormFunc(planes)
          self.prelu = nn.PReLU()
        self.bottle_neck = bottle_neck
        self.downsample = downsample
        self.use_se = use_se
        if self.use_se:
          if bottle_neck:
            self.se = SEModule(planes * 4)
          else:
            self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        if self.bottle_neck:
          out = self.norm1(x)
          out = self.conv1(out)
          out = self.norm2(out)
          out = self.prelu(out)

          out = self.conv2(out)
          out = self.norm3(out)
          out = self.prelu(out)
          out = self.conv3(out)
          out = self.norm4(out)
        else:
          out = self.norm1(x)
          out = self.conv1(out)
          out = self.norm2(out)
          out = self.prelu(out)

          out = self.conv2(out)
          out = self.norm3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

class SEModule(nn.Module):
  def __init__(self, channels, reduction = 16):
    super(SEModule, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Conv2d(
      channels, channels // reduction, kernel_size=1, padding=0, bias=False)
    self.pelu = nn.PReLU()
    self.fc2 = nn.Conv2d(
      channels // reduction, channels, kernel_size=1, padding=0, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    module_input = x
    x = self.avg_pool(x)
    x = self.fc1(x)
    x = self.pelu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return module_input * x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_layers, use_se=True):
        self.use_se = use_se
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.norm1 = NormFunc(64)
        self.prelu = nn.PReLU()
        if num_layers >= 101:
          self.layer1 = self._make_layer(block, 256, layers[0], stride=2, bottle_neck=True)
          self.layer2 = self._make_layer(block, 512, layers[1], stride=2, bottle_neck=True)
          self.layer3 = self._make_layer(block, 1024, layers[2], stride=2, bottle_neck=True)
          self.layer4 = self._make_layer(block, 2048, layers[3], stride=2, bottle_neck=True)
        else:
          self.layer1 = self._make_layer(block, 64, layers[0], stride=2, bottle_neck=False)
          self.layer2 = self._make_layer(block, 128, layers[1], stride=2, bottle_neck=False)
          self.layer3 = self._make_layer(block, 256, layers[2], stride=2, bottle_neck=False)
          self.layer4 = self._make_layer(block, 512, layers[3], stride=2, bottle_neck=False)

        self.output_layer = nn.Sequential(NormFunc(512 if num_layers < 101 else 2048),
                                       nn.Dropout(0.4),
                                       Flatten(),
                                       nn.Linear((512 if num_layers < 101 else 2048) * 7 * 7, 512),
                                       nn.BatchNorm1d(512))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight)
                scale = math.sqrt(3. / m.in_features)
                m.weight.data.uniform_(-scale, scale)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, bottle_neck = False):
        downsample = None
        if bottle_neck:
          block.expansion = 4
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                NormFunc(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, bottle_neck, stride, downsample, use_se=self.use_se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bottle_neck=bottle_neck, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return F.normalize(x, p=2, dim=1)


def resnet18(use_se = False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(IRBlock, [2, 2, 2, 2], num_layers = 18, use_se = use_se)
    return model


def resnet34(use_se = False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(IRBlock, [3, 4, 6, 3], num_layers = 34, use_se = use_se)
    return model


def resnet50(use_se = False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(IRBlock, [3, 4, 14, 3], num_layers = 50, use_se = use_se)
    return model


def resnet100(use_se=False):
    """Constructs a ResNet-100 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(IRBlock, [3, 13, 30, 3], num_layers = 100, use_se = use_se)
    return model

def resnet101(use_se = False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(IRBlock, [3, 4, 23, 3], num_layers = 101, use_se = use_se)
    return model


def resnet152(use_se = False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(IRBlock, [3, 8, 36, 3],  num_layers = 152, use_se = use_se)
    return model
