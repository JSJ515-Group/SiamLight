import torch
import torch.nn as nn
import math  
from siamlight.core.xcorr import xcorr_fast, xcorr_depthwise, pg_xcorr
import torch.nn.functional as F
from siamlight.core.config import cfg
class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4 

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module): 
    def __init__(self, in_channels,hidden,  out_channels,kernel_size=3):
        super(DepthwiseXCorr, self).__init__()

        self.conv_kernel = nn.Sequential(
                # dw 
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(out_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
                )
        
        self.conv_search = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=in_channels,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                # pw 
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                ) 
        #initialization这些层里面的卷积参数都进行初始化
        for modules in [self.conv_kernel, self.conv_search]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel) 
        search = self.conv_search(search) 
        feature = xcorr_depthwise(search, kernel)
        return feature

class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=48, reduction=1,kernel_size=7,padding=3):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(48, 2, kernel_size, padding=padding, bias=False)
        #self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
        #                     padding=0)
        self.relu = nn.ReLU(inplace=True)
        #self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
        #                      padding=0)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return module_input * x


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
class CBAM(nn.Module):
    def __init__(self, channels=48, reduction=1, kernel_size=7,padding=3):
        super(CBAM, self).__init__()
        self.ca = CAModule(channels, reduction)
        #self.sa = SpatialAttention(kernel_size)
        #self.CA_layer = CAModule(channels=48)
        #self.conv1 = nn.Conv2d(48, 48, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        out = x * self.ca(x)
        #result = out * self.sa(out)
        #result1 =self.conv1(result)
        return out



class PixelwiseXCorr(nn.Module):  #in_channels:48  out_channels:48
    def __init__(self, in_channels, out_channels,kernel_size=3):
        super(PixelwiseXCorr, self).__init__()

        self.CA_layer = CBAM(channels=48)
        #self.CA_layer = CAModule(channels=48)

    def forward(self, kernel, search):

        feature = pg_xcorr(search,kernel) #使用pixel-to-global correltion

        corr = self.CA_layer(feature)

        return corr

#不改变头部
class DepthwiseBAN(BAN):
    def __init__(self, in_channels=48, out_channels=48, weighted=False):
        super(DepthwiseBAN, self).__init__()

        self.cls_dw = PixelwiseXCorr(in_channels, out_channels)
        self.reg_dw = PixelwiseXCorr(in_channels, out_channels)

        cls_tower = []
        bbox_tower = []

        # （1）特征增强网络
        for i in range(cfg.TRAIN.NUM_CONVS):  #
            cls_tower.append(nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            cls_tower.append(nn.GroupNorm(48, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ))
            bbox_tower.append(nn.GroupNorm(48, in_channels))
            bbox_tower.append(nn.ReLU())

        # add_module 函数，为module添加一个子module函数  https://blog.csdn.net/qq_31964037/article/details/105416291
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )

        # （3）回归分支输出  input:[256] --> output:[4]
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )

        #  （5）权重初
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, z_f, x_f):

        x_cls = self.cls_dw(z_f, x_f)

        x_reg = self.reg_dw(z_f, x_f)

        # head-分类
        cls_tower = self.cls_tower(x_cls)  # [B, 256, 25, 25] --> [B, 256, 25, 25]

        logits = self.cls_logits(cls_tower)  # [B, 256, 25, 25] --> [B, 2, 25, 25]

        # head-回归
        bbox_tower = self.bbox_tower(x_reg)

        bbox_reg = self.bbox_pred(bbox_tower)  # [B, 256, 25, 25] --> [B, 4, 25, 25]

        bbox_reg = torch.exp(bbox_reg)  # [B, 4, 25, 25] --> [B, 4, 25, 25]

        return logits, bbox_reg
