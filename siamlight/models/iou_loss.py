import torch
import math
from torch import nn

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None,eps=1e-7):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        #预测框、真实框

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'diou':
            # compute iou
            dist = torch.square((pred_left - target_left + target_right - pred_right) / 2) + \
                   torch.square((pred_top - target_top + target_bottom - pred_bottom) / 2)
            x_in_1 = torch.min(-pred_left, -target_left)
            y_in_1 = torch.min(-pred_top, -target_top)
            x_in_2 = torch.min(pred_right, target_right)
            y_in_2 = torch.min(pred_bottom, target_bottom)
            #返回带有输入元素平方的新张量
            c = torch.square(x_in_1 - x_in_2) + torch.square(y_in_1 - y_in_2)
            dious = ious - dist / c
            #clamp函数的功能是将输入input张量每个元素的值压缩到区间[min,max],并返回结果到一个新张量
            dious = torch.clamp(dious, min=-1.0, max=1.0)

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
            if losses.numel() == 0:
                losses = torch.zeros((50))
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
            if losses.numel() == 0:
                losses = torch.zeros((50))
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious

        elif self.loc_loss_type == "diou":
            losses = 1- dious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


linear_iou = IOULoss(loc_loss_type='iou')
