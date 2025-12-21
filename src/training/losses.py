"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the true class.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights (tensor of shape (num_classes,)) or None
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Args:
            inputs: Logits of shape (B, C)
            targets: Ground truth labels of shape (B,)

        Returns:
            Focal loss
        """
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal term
        focal_term = (1 - p_t) ** self.gamma

        # Compute focal loss
        loss = focal_term * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def build_loss_fn(config, class_weights=None):
    """
    Build loss function from config.

    Args:
        config: Configuration object
        class_weights: Optional class weights tensor

    Returns:
        Loss function
    """
    loss_name = config.get("training.loss.name", "FocalLoss")

    if loss_name == "FocalLoss":
        gamma = config.get("training.loss.focal_gamma", 2.0)

        # Use provided class weights or compute from config
        if class_weights is not None:
            alpha = class_weights
        else:
            alpha = None

        return FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')

    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(weight=class_weights)

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
