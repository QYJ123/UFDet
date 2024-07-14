import torch
import math
from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

pi = math.pi

@BBOX_CODERS.register_module()
class P5OffsetCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        encoded_bboxes = bbox2delta_sp(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta_sp2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                       wh_ratio_clip)
        
        return decoded_bboxes


def bbox2delta_sp(proposals, gt,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.)):
    assert proposals.size(0) == gt.size(0)

    proposals = proposals.float()
    gt = gt.float()
    
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    
    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2]
    gh = gt[..., 3]
    gtheta = gt[..., 4]   
    gw_regular = torch.where(gtheta.abs()<= pi/4, gw, gh)
    gh_regular = torch.where(gtheta.abs()<= pi/4, gh, gw)
    dtheta_regular = torch.where(gtheta.abs()<= pi/4, gtheta, (gtheta-pi/2)*(gtheta>=0)+(gtheta+pi/2)*(gtheta<0))
    dx = (gx-px) / (pw**2+ph**2).sqrt()
    dy = (gy-py) / (pw**2+ph**2).sqrt()
    
    dL = ((gw_regular-gh_regular)*dtheta_regular.cos()+(gw_regular+gh_regular)*dtheta_regular.sin()-(pw-ph))/(pw**2+ph**2).sqrt()
    da = ((gw_regular-gh_regular)*dtheta_regular.cos()-(gw_regular+gh_regular)*dtheta_regular.sin()-(pw-ph))/(pw**2+ph**2).sqrt()
    db = ( ((gw_regular-gh_regular)*dtheta_regular.sin()).abs()-(gw_regular+gh_regular)*dtheta_regular.cos()+(pw+ph))/(pw**2+ph**2).sqrt()

    deltas = torch.stack([dx, dy, dL, da, db], dim=-1)    
    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)
    return deltas


def delta_sp2bbox(rois, deltas,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx =  denorm_deltas[:, 0::5]
    dy =  denorm_deltas[:, 1::5]
    dL0 = denorm_deltas[:, 2::5]
    da0 = denorm_deltas[:, 3::5]
    db0 = denorm_deltas[:, 4::5]
     
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dx)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dx)
    ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dx)
    
    dL1 = dL0+pw/(pw**2+ph**2).sqrt()
    da1 = da0+pw/(pw**2+ph**2).sqrt()
    db1 = db0-ph/(pw**2+ph**2).sqrt()
 
    db = (db1-torch.min(dL,da)).clamp(max=0)+torch.min(dL,da)   

    gx = dx*(pw**2+ph**2).sqrt() + px
    gy = dy*(pw**2+ph**2).sqrt() + py
    
    A = (db**2+(dL**2-da**2).abs()).sqrt()
    dthetap = torch.atan2(dL-da,A-db)
    
    wp_add_hp = 0.5*(pw**2+ph**2).sqrt()*((dL-da)**2+(A-db)**2).sqrt()
    wp_sub_hp = 0.5*(pw**2+ph**2).sqrt()*((dL+da)**2+(A+db)**2).sqrt()*(dL+da).sign()
    wp = 0.5*(wp_add_hp+wp_sub_hp).abs()
    hp = 0.5*(wp_add_hp-wp_sub_hp).abs()
    thetap1 = dthetap.clamp(min=-pi/4,max=pi/4) 
    wp_regular = torch.where(wp>=hp,wp,hp)
    hp_regular = torch.where(wp>=hp,hp,wp)
    thetap_regular = torch.where(wp>=hp,thetap1,(thetap1-pi/2)*(thetap1>=0)+(thetap1+pi/2)*(thetap1<0))
    bboxes = torch.stack([gx, gy, wp_regular, hp_regular, thetap_regular], dim=-1)
    return bboxes.view_as(deltas)
