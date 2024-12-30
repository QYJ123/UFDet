import torch
import math
from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

pi = math.pi

@BBOX_CODERS.register_module()
class OBB2OBBMDeltaXYWABCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert gt_bboxes.size(-1) == bboxes.size(-1) 
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
    
    dtheta0 = gtheta-ptheta
    #dtheta1 = torch.where(dtheta0>=pi/2,dtheta0-pi,torch.where(dtheta0<-pi/2,dtheta0+pi,dtheta0)) 
    dtheta1 = torch.where(dtheta0.abs()<=pi/2,dtheta0,(dtheta0-pi)*(dtheta0>0)+(dtheta0+pi)*(dtheta0<0)) 
    #sign_1 = (dtheta1.abs()-pi/2)*(gw/gh-1)  

    gw_regular = torch.where(4*dtheta1.abs()<=pi, gw, gh)
    gh_regular = torch.where(4*dtheta1.abs()<=pi, gh, gw)
    dtheta_regular = torch.where(4*dtheta1.abs()<=pi,dtheta1, (dtheta1-pi/2)*(dtheta1>=0)+(dtheta1+pi/2)*(dtheta1<0))

    va = ((gw_regular**2+gh_regular**2).sqrt()*((gw_regular**2+gh_regular**2).sqrt()-gh_regular)/2).sqrt()
    vb = ((gw_regular**2+gh_regular**2).sqrt()*((gw_regular**2+gh_regular**2).sqrt()+gh_regular)/2).sqrt()
    v1g = ((pw**2+ph**2).sqrt()*((pw**2+ph**2).sqrt()-ph)/2).sqrt()
    v2g = ((pw**2+ph**2).sqrt()*((pw**2+ph**2).sqrt()+ph)/2).sqrt()

    dx = (torch.cos(-ptheta)*(gx-px)+torch.sin(-ptheta)*(gy-py)) /(pw**2+ph**2).sqrt()
    dy = (-torch.sin(-ptheta)*(gx-px)+torch.cos(-ptheta)*(gy-py))/(pw**2+ph**2).sqrt()
    dL =(va*dtheta_regular.cos()+vb*dtheta_regular.sin()-v1g)/(pw**2+ph**2).sqrt()
    da =(va*dtheta_regular.cos()-vb*dtheta_regular.sin()-v1g)/(pw**2+ph**2).sqrt()
    db =(va*dtheta_regular.sin().abs()-vb*dtheta_regular.cos()+v2g)/(pw**2+ph**2).sqrt()
    
    deltas = torch.stack([dx, dy, dL, da, db], dim=-1)    
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

    px, py, pw, ph, ptheta = proposals.unbind(dim=-1)
    px = px.unsqueeze(1).expand_as(dx)
    py = py.unsqueeze(1).expand_as(dy)
    pw = pw.unsqueeze(1).expand_as(dx)
    ph = ph.unsqueeze(1).expand_as(dy)
    ptheta = ptheta.unsqueeze(1).expand_as(dx)
    
    vg1 = ((pw**2+ph**2).sqrt()*((pw**2+ph**2).sqrt()-ph)/2).sqrt()
    vg2 = ((pw**2+ph**2).sqrt()*((pw**2+ph**2).sqrt()+ph)/2).sqrt()

    dx  = denorm_deltas[:, 0::5]
    dy  = denorm_deltas[:, 1::5]
    dL = denorm_deltas[:, 2::5]*(pw**2+ph**2).sqrt()+vg1
    da = denorm_deltas[:, 3::5]*(pw**2+ph**2).sqrt()+vg1
    db = denorm_deltas[:, 4::5]*(pw**2+ph**2).sqrt()-vg2
    
    gx = (dx*torch.cos(-ptheta) - dy*torch.sin(-ptheta))*(pw**2+ph**2).sqrt() + px
    gy = (dx*torch.sin(-ptheta) + dy*torch.cos(-ptheta))*(pw**2+ph**2).sqrt() + py
    
    A = (db**2+(dL**2-da**2).abs()).sqrt()
    dthetap = torch.atan2(dL-da,A-db)
    
    v1 = 0.5*((dL+da)**2+(A+db)**2)
    v2 = 0.5*((dL-da)**2+(A-db)**2)
    wp = (2*v1*v2/(v1+v2)).sqrt()
    hp = ((v2-v1)**2/(2*(v1+v2))).sqrt()

    gtheta0 = ptheta+dthetap.clamp(min=-pi/4,max=pi/4)  
    #thetap1 = torch.where(gtheta0>pi/2,gtheta0-pi,(gtheta0+pi)*(gtheta0<-pi/2)+gtheta0*(gtheta0>=-pi/2)*(gtheta0<pi/2))
    thetap1 = torch.where(gtheta0.abs()<=pi/2,gtheta0,(gtheta0+pi)*(gtheta0<0)+(gtheta0-pi)*(gtheta0>0))
    wp_regular = torch.where(wp>=hp,wp,hp)
    hp_regular = torch.where(wp>=hp,hp,wp)
    thetap_regular = torch.where(wp>=hp,thetap1,(thetap1-pi/2)*(thetap1>=0)+(thetap1+pi/2)*(thetap1<0))
    bboxes = torch.stack([gx, gy, wp_regular, hp_regular, thetap_regular], dim=-1)
    return bboxes.view_as(deltas)
