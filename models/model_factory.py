from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as torch_models
from models import resnet_sn

model_names = sorted(name for name in torch_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torch_models.__dict__[name]))

def get_model(name, **kwargs):

    if name in model_names:
        model = torch_models.__dict__[name]()
        return model

    custom_model_map = {
        'resnet_v1_sn_50': resnet_sn.resnet50(**kwargs),
        'resnet_v1_sn_101': resnet_sn.resnet101(**kwargs),
    }
    if name in custom_model_map:
        model = custom_model_map[name]
        return model

    if name not in custom_model_map or model_names:
        raise ValueError('Model name [%s] was not recognized' % name)



