# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


#from nanotrack.models.backbone.mobile_v3 import mobilenetv3_small
from nanotrack.models.backbone.mobile_v3 import mobilenetv3_large

BACKBONES = {
              #'mobilenetv3_small': mobilenetv3_small,
              'mobilenetv3_large': mobilenetv3_large,

}

def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs) 
