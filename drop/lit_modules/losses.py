import torch
import torch.nn as nn
import torchvision
from torch import Tensor

class FocalLossBCE(nn.Module):
    """Focal loss for binary classification.
    Used in RetinaNet paper by Lin et al.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """ "
        If alpha is equal to -1 and gamma equal to 0.0, then this loss equals CrossEntropyLoss.
        Parameters
        ----------
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (str): Specifies the reduction to apply to the output:
        """
        super(FocalLossBCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """ The target tensor needs to be of the same dtype as the input tensor.
        If there are multiple classespredicted, take the second class (for binary prediction)."""
        if input.shape[-1] > 1:
            input = input[:, 1]
        return torchvision.ops.sigmoid_focal_loss(
            inputs=input,
            targets=target.type(torch.float),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class FocalLossCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
         If alpha is equal to -1 and gamma equal to 0.0, then this loss equals CrossEntropyLoss.
         The CrossEntropyLossApplies the Softmax followed by NLLLoss to the input.
        Parameters
        ----------
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``. Default is 0.25 which downweights the loss for the negative class.
                This is appropriate for object detection.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (str): Specifies the reduction to apply to the output:
        """
        super(FocalLossCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_loss = nn.CrossEntropyLoss(reduce=False)(input, target)

        pt = torch.exp(-ce_loss)
        focal_loss = ce_loss * ((1 - pt) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            focal_loss = focal_loss.mean()
        elif self.reduction == "sum":
            focal_loss = focal_loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return focal_loss
