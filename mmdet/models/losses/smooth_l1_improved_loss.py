import torch
import torch.nn as nn
import math
from ..builder import LOSSES
from .utils import weighted_loss

from .smooth_l1_loss import smooth_l1_loss,l1_loss

@weighted_loss
def sqrt_loss(pred,target):
    """L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = ((-2*pred[:,2]).exp()*(pred[:,0]-target[:,0])**2+(-2*pred[:,3]).exp()*(pred[:,1]-target[:,1])**2).sqrt()
    #loss = ((pred[:,0]-target[:,0])**2+(pred[:,1]-target[:,1])**2)
    return loss

@weighted_loss
def decrease_one_loss(pred,target,anchors):
    """L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    anchors = anchors

    pw = (anchors[:, 2]-anchors[:, 0])
    ph = (anchors[:, 3]-anchors[:, 1])
    pr = pw/ph
    assert pred.size() == target.size() and target.numel() > 0
    
    q = ((pr**2*(2*pred[:,0]-2*pred[:,1]).exp()+4*pred[:,3]**2)/(1+4*pred[:,2]**2*pr**2*(2*pred[:,0]-2*pred[:,1]).exp())).sqrt()
    k = torch.where(q>=1,q,1/q) 
    
    #loss_area =( (pred[:,0]+pred[:,1]-target[:,0]-target[:,1])+(pred[:,2]*pred[:,3]-target[:,2]*target[:,3])+k.log() )**2
    #loss_area =0.5*(pred[:,0]+pred[:,1]-target[:,0]-target[:,1])**2+(pred[:,2]*pred[:,3]-target[:,2]*target[:,3])**2+(k.log())**2
    loss_area = (pred[:,2]*pred[:,3]-target[:,2]*target[:,3])**2+(k.log())**2
    loss = loss_area/2
    
    return loss


@LOSSES.register_module()
class SmoothL1IMPROVEDLoss(nn.Module):
    """Smooth L1 loss

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1IMPROVEDLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                anchors=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_center = 0.1*self.loss_weight*sqrt_loss(
            pred[:,0:4],
            target[:,0:4],
            weight[:,0:4].mean(dim=1),
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        loss_area =  0.1*self.loss_weight*decrease_one_loss(
            pred[:,2:6],
            target[:,2:6],
            weight[:,2:6].mean(dim=1), 
            anchors=anchors,  
            reduction=reduction,
            avg_factor=avg_factor,**kwargs)

        loss_bbox = 0.8*self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss_bbox+loss_center+loss_area




