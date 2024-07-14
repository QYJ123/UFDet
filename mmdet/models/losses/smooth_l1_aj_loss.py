import torch
import torch.nn as nn
import math
from ..builder import LOSSES
from .utils import weighted_loss

from .smooth_l1_loss import smooth_l1_loss,l1_loss
def s_m(x,beta=1):
    diff = x.abs()
    y = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return y

@weighted_loss
def center_loss(pred,target,beta=1.0):
    """L1 loss

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    e_center = ((pred[:,0]-target[:,0])**2+(pred[:,1]-target[:,1])**2).sqrt()
    loss = 0.5*e_center**2*(e_center<=1)+(e_center.abs()-0.5)*(e_center>1)
    #loss = s_m(e_center,beta)
    return loss


@weighted_loss
def D_loss(pred,target,beta=1.0):
    
    assert pred.size() == target.size() and target.numel() > 0
    
    dLp = pred[:,0]
    dap = pred[:,1]
    dbp = pred[:,2]
    Ap = (dbp**2+(dLp**2-dap**2).abs()).sqrt()

    dag = target[:,0]
    dLg = target[:,1]
    dbg = target[:,2]
    Ag = (dbg**2+(dLg**2-dag**2).abs()).sqrt()
    
    e_A = Ap-Ag
    loss_LD = e_A.abs()
    e_D = (dLp**2+dap**2+Ap**2+dbp**2).sqrt()-(dLg**2+dag**2+Ag**2+dbg**2).sqrt()
    loss_D = e_D.abs()
    loss = 0.5*loss_LD+0.1*loss_D
    return loss


@LOSSES.register_module()
class SmoothL1AJLoss(nn.Module):
    """Smooth L1 loss

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1AJLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
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

        loss_D = self.loss_weight*D_loss(
            pred[:,2:5],
            target[:,2:5],
            weight[:,2:5].mean(dim=1),
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        loss_center = self.loss_weight*center_loss(
            pred[:,0:2],
            target[:,0:2],
            weight[:,0:2].mean(dim=1),
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred[:,2:5],
            target[:,2:5],
            weight[:,2:5],
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        total_loss = loss_bbox+loss_center+loss_D

        return total_loss

