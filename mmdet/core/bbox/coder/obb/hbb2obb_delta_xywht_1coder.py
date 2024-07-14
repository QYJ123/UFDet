import numpy as np
import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import regular_theta, regular_obb
from mmdet.core.bbox.builder import BBOX_CODERS
import pdb

pi = 3.141592


@BBOX_CODERS.register_module()
class HBB2OBBDeltaXYWHT1Coder(BaseBBoxCoder):

    def __init__(self,
                 theta_norm=True,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.theta_norm = theta_norm
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        encoded_bboxes = obb2delta(bboxes, gt_bboxes, self.theta_norm, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16/1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2obb(bboxes, pred_bboxes, self.theta_norm, 
                                   self.means, self.stds, wh_ratio_clip)

        return decoded_bboxes


def obb2delta(proposals, gt, theta_norm=True, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    gx, gy, gw, gh, gtheta = gt.unbind(dim=-1)

    dtheta1 = regular_theta(gtheta)
    dtheta2 = regular_theta(gtheta + pi/2)
    abs_dtheta1 = torch.abs(dtheta1)
    abs_dtheta2 = torch.abs(dtheta2)

    gw_regular = torch.where(abs_dtheta1 < abs_dtheta2, gw, gh)
    gh_regular = torch.where(abs_dtheta1 < abs_dtheta2, gh, gw)
    dtheta_regular = torch.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
    dx = (gx-px) / (pw**2+ph**2).sqrt()
    dy = (gy-py) / (pw**2+ph**2).sqrt()
    
    dL = (gw_regular*dtheta_regular.cos()+gh_regular*dtheta_regular.sin())/(pw**2+ph**2).sqrt()
    da = (gw_regular*dtheta_regular.cos()-gh_regular*dtheta_regular.sin())/(pw**2+ph**2).sqrt()
    db = (gw_regular*dtheta_regular.sin().abs()-gh_regular*dtheta_regular.cos())/(pw**2+ph**2).sqrt()

    deltas = torch.stack([dx, dy, dL, da, db], dim=-1)    
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def delta2obb(proposals,
              deltas,
              theta_norm=True,
              means=(0., 0., 0., 0., 0.),
              stds=(1., 1., 1., 1., 1.),
              wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx =  denorm_deltas[:, 0::5]
    dy =  denorm_deltas[:, 1::5]
    dL0 = denorm_deltas[:, 2::5]
    da0 = denorm_deltas[:, 3::5]
    db0 = denorm_deltas[:, 4::5]

    px = ((proposals[:, 0] + proposals[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((proposals[:, 1] + proposals[:, 3]) * 0.5).unsqueeze(1).expand_as(dx)
    pw = (proposals[:, 2] - proposals[:, 0]).unsqueeze(1).expand_as(dx)
    ph = (proposals[:, 3] - proposals[:, 1]).unsqueeze(1).expand_as(dx)

    da = (dL0+da0).clamp(min=0)-dL0
    dL = (dL0+da0).clamp(min=0)-da0  
    db = (db0-torch.min(dL,da)).clamp(max=0)+torch.min(dL,da) 

    gx = dx*(pw**2+ph**2).sqrt() + px
    gy = dy*(pw**2+ph**2).sqrt() + py
    
    A = (db**2+(dL**2-da**2).abs()).sqrt()
    dthetap = torch.atan2(dL-da,A-db)
    
    hp = 0.5*(pw**2+ph**2).sqrt()*((dL-da)**2+(A-db)**2).sqrt()
    wp = 0.5*(pw**2+ph**2).sqrt()*((dL+da)**2+(A+db)**2).sqrt()
    thetap1 = dthetap.clamp(min=-pi/4,max=pi/4) 
    wp_regular = torch.where(wp>=hp,wp,hp)
    hp_regular = torch.where(wp>=hp,hp,wp)
    thetap_regular = torch.where(wp>=hp,thetap1,(thetap1-pi/2)*(thetap1>=0)+(thetap1+pi/2)*(thetap1<0))
    bboxes = torch.stack([gx, gy, wp_regular, hp_regular, thetap_regular], dim=-1)
    return bboxes.view_as(deltas)
