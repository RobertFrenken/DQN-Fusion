import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hydra
from sklearn.utils.class_weight import compute_class_weight

#########################################
# 0) Distillation Loss Function         #
#########################################
def distillation_loss_fn(student_logits, teacher_logits, T=5.0):
    """
    Compute the distillation loss with temperature scaling.
    """
    # Clamp logits to prevent numerical instability
    teacher_logits = torch.clamp(teacher_logits, -10, 10)
    student_logits = torch.clamp(student_logits, -10, 10)

    # Compute softmax probabilities with temperature scaling
    teacher_prob = F.softmax(teacher_logits / T, dim=-1)
    student_log_prob = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence loss
    distill_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T * T)
    return distill_loss
#########################################
# 0) Focal Loss Function         #
#########################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance.
        Args:
            alpha (float): Weighting factor for the minority class.
            gamma (float): Focusing parameter to reduce the loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Compute the probability of the correct class
        pt = torch.exp(-BCE_loss)
        # Apply the focal loss formula
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        

def compute_class_weights(labels):
    """
    Compute class weights based on the distribution of labels.
    Args:
        labels (numpy array): Array of binary labels (0 and 1).
    Returns:
        torch.Tensor: Class weights for the minority and majority classes.
    """
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)