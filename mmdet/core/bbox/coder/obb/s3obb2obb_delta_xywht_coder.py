import numpy as np
import torch
import math
from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import regular_theta, regular_obb
from mmdet.core.bbox.builder import BBOX_CODERS

pi = math.pi


@BBOX_CODERS.register_module()
class S3OBB2OBBDeltaXYWHTCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert gt_bboxes.size(-1) == bboxes.size(-1) == 5
        encoded_bboxes = obb2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16/1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2obb(bboxes, pred_bboxes, self.means, self.stds,
                                   wh_ratio_clip)

        return decoded_bboxes


def obb2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    proposals = proposals.float()
    gt = gt.float()
    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)
    gx, gy, gw, gh, gtheta = gt.unbind(dim=-1)

    
    dtheta1 = regular_theta(gtheta-ptheta)
    dtheta2 = regular_theta(gtheta-ptheta + pi/2)
    abs_dtheta1 = torch.abs(dtheta1)
    abs_dtheta2 = torch.abs(dtheta2)

    gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
    gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
    dtheta = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
    dx = (torch.cos(-ptheta)*(gx-px)+torch.sin(-ptheta)*(gy-py)) / pw
    dy = (-torch.sin(-ptheta)*(gx-px)+torch.cos(-ptheta)*(gy-py)) / ph
    dw = torch.log(gw_regular/pw)
    dh = torch.log(gh_regular/ph)

    ds = torch.log(gw_regular*gh_regular / pw/ph)
    dr = torch.log((gw/gh)/ (pw/ph))

    da = 0.25*torch.sin(2*dtheta)*(gw_regular/gh_regular+gh_regular/gw_regular)
    deltas = torch.stack([dx, dy, ds, dr, da], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def delta2obb(proposals,
              deltas,
              means=(0., 0., 0., 0., 0.),
              stds=(1., 1., 1., 1., 1.),
              wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    ds = denorm_deltas[:, 2::5]
    dr = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    
    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)

    px = px.unsqueeze(1).expand_as(dx)
    py = py.unsqueeze(1).expand_as(dy)
    pw = pw.unsqueeze(1).expand_as(dx)
    ph = ph.unsqueeze(1).expand_as(dy)
    ptheta = ptheta.unsqueeze(1).expand_as(da)
    
    
    gx = dx*pw*torch.cos(-ptheta) - dy*ph*torch.sin(-ptheta) + px
    gy = dx*pw*torch.sin(-ptheta) + dy*ph*torch.cos(-ptheta) + py
    
    
    dw = (ds+dr)/2
    dh = (ds-dr)/2
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    gw = pw*dw.exp()
    gh = ph*dh.exp()


    delta_e = (da*4/(gw/gh+gh/gw)).clamp(min=-1, max=1).asin()
   
    gtheta = delta_e/2 + ptheta
    
    gw_regular = torch.where(gw >= gh, gw, gh)
    gh_regular = torch.where(gw >= gh, gh, gw)
    gtheta_regular = torch.where(gw >= gh, gtheta,gtheta-math.pi/2*(gtheta+1e-8>=0)+math.pi/2*(gtheta+1e-8<0))
    bboxes = torch.cat([gx, gy, gw_regular, gh_regular, gtheta_regular], dim=1)
    return bboxes.view_as(deltas)
