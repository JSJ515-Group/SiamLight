from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nanotrack.models.head.ban import UPChannelBAN, DepthwiseBAN



BANS = {
        'UPChannelBAN': UPChannelBAN,
        'DepthwiseBAN': DepthwiseBAN,  #这就是一个rpn头，采用这个的话，就是属于rpn内容

       }


def get_ban_head(name, **kwargs):
    return BANS[name](**kwargs)
