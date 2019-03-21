import torch.nn as nn
from .backbones import resnet
from .head import ArcFullyConnected
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))
from devkit.ops import SyncSwitchableNorm2d, SwitchNorm2d, SyncBatchNorm2d



class ArcFaceWithLoss(nn.Module):
  def __init__(self, backbone, num_classes, norm_func = 'bn', feature_dim = 512, use_se = False):
    super(ArcFaceWithLoss, self).__init__()

    if norm_func == 'bn':
      def NormFunc(*args, **kwargs):
        return nn.BatchNorm2d(*args, **kwargs, eps=2e-5, momentum=0.9)
    elif norm_func == 'syncbn':
      def NormFunc(*args, **kwargs):
        return SyncBatchNorm2d(*args, **kwargs, eps=2e-5, momentum=0.9)
    elif norm_func == 'sn':
      def NormFunc(*args, **kwargs):
        return SwitchNorm2d(*args, **kwargs, eps=2e-5, momentum=0.9)
    elif norm_func == 'syncsn':
      def NormFunc(*args, **kwargs):
        return SyncSwitchableNorm2d(*args, **kwargs, eps=2e-5, momentum=0.9)
    resnet.NormFunc = NormFunc
    self.basemodel = resnet.__dict__[backbone](use_se = use_se)
    self.head = ArcFullyConnected(feature_dim, num_classes)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, input, target=None, extract_mode=False):

    feature = self.basemodel(input)
    if extract_mode:
      return feature
    else:
      thetas = self.head(feature, target)
      loss = self.criterion(thetas, target)
      return loss