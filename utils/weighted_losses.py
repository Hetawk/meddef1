import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from collections import Counter


class WeightedLoss:
    """
    A collection of weighted loss functions for handling class imbalance and difficult examples.
    """

    @staticmethod
    def get_class_weights(targets, method='inverse', scaling='log', smooth_factor=0.1):
        """
        Calculate class weights based on class distribution.

        Args:
            targets (torch.Tensor or list): Class labels
            method (str): Weighting method: 'inverse', 'squared_inverse', or 'effective_samples'
            scaling (str): Scaling method: 'none', 'log', or 'sqrt'
            smooth_factor (float): Smoothing factor to avoid extreme weights

        Returns:
            torch.Tensor: Class weights tensor
        """
        # Convert targets to list if it's a tensor
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy().tolist()
        elif hasattr(targets, 'targets'):
            # Handle dataset with targets attribute
            targets = targets.targets

        # Count class frequencies
        class_counts = Counter(targets)
        num_samples = len(targets)
        num_classes = len(class_counts)

        # Get frequencies in order of class indices
        frequencies = np.array([class_counts.get(i, 0)
                               for i in range(max(class_counts.keys()) + 1)])

        # Add smoothing to avoid division by zero
        frequencies = frequencies + smooth_factor

        # Calculate weights based on method
        if method == 'inverse':
            weights = num_samples / (num_classes * frequencies)
        elif method == 'squared_inverse':
            weights = (num_samples / (num_classes * frequencies)) ** 2
        elif method == 'effective_samples':
            beta = (num_classes - 1) / num_classes
            effective_num = 1.0 - np.power(beta, frequencies)
            weights = (1.0 - beta) / np.array(effective_num)
        else:
            raise ValueError(f"Unsupported weighting method: {method}")

        # Apply scaling
        if scaling == 'log':
            weights = 1 + np.log(weights)
        elif scaling == 'sqrt':
            weights = np.sqrt(weights)

        # Normalize weights
        weights = weights / weights.sum() * num_classes

        logging.info(
            f"Class weights calculated ({method}, {scaling}): {weights}")
        return torch.FloatTensor(weights)

    @staticmethod
    def weighted_cross_entropy_loss(outputs, targets, weights=None, dataset=None, smooth_factor=0.1):
        """
        Weighted cross-entropy loss that automatically detects and applies class weighting.

        Args:
            outputs (torch.Tensor): Model outputs/logits
            targets (torch.Tensor): Target labels
            weights (torch.Tensor, optional): Pre-computed class weights
            dataset (torch.utils.data.Dataset, optional): Dataset to extract class distribution
            smooth_factor (float): Smoothing factor for weight calculation

        Returns:
            torch.Tensor: Weighted loss value
        """
        # If weights not provided, calculate them
        if weights is None:
            if dataset is not None:
                # Use dataset if available
                all_targets = dataset
            else:
                # Otherwise use current batch targets
                all_targets = targets

            weights = WeightedLoss.get_class_weights(
                all_targets, method='inverse', scaling='log', smooth_factor=smooth_factor)

        # Move weights to the same device as targets
        weights = weights.to(targets.device)

        # Apply weighted cross entropy loss
        return F.cross_entropy(outputs, targets, weight=weights)

    @staticmethod
    def aggressive_minority_weighted_loss(outputs, targets, dataset=None, smooth_factor=0.1):
        """
        Applies squared inverse frequency with log scaling for more aggressive minority class weighting.

        Args:
            outputs (torch.Tensor): Model outputs/logits
            targets (torch.Tensor): Target labels
            dataset (torch.utils.data.Dataset, optional): Dataset to extract class distribution
            smooth_factor (float): Smoothing factor for weight calculation

        Returns:
            torch.Tensor: Weighted loss value
        """
        # Calculate weights with squared inverse frequency and log scaling
        if dataset is not None:
            all_targets = dataset
        else:
            all_targets = targets

        weights = WeightedLoss.get_class_weights(
            all_targets, method='squared_inverse', scaling='log', smooth_factor=smooth_factor)

        # Move weights to the same device as targets
        weights = weights.to(targets.device)

        return F.cross_entropy(outputs, targets, weight=weights)

    @staticmethod
    def dynamic_sample_weighting(outputs, targets, epoch, max_epochs, alpha=0.5, gamma=2.0):
        """
        Focal loss with dynamic alpha parameter that increases focus on hard examples over training time.

        Args:
            outputs (torch.Tensor): Model outputs/logits
            targets (torch.Tensor): Target labels  
            epoch (int): Current epoch
            max_epochs (int): Maximum number of epochs
            alpha (float): Weighting factor for focal loss
            gamma (float): Focusing parameter for focal loss

        Returns:
            torch.Tensor: Weighted loss value
        """
        # Calculate dynamic gamma that increases with epochs
        progress_ratio = min(epoch / max_epochs, 1.0)
        dynamic_gamma = gamma * (1 + progress_ratio)

        # Calculate probabilities
        probs = F.softmax(outputs, dim=1)
        probs_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal loss calculation
        focal_weight = (1 - probs_t) ** dynamic_gamma

        # Standard cross-entropy calculation
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Apply focal weighting
        weighted_loss = focal_weight * loss

        # Apply alpha balancing if needed
        if alpha is not None:
            # Can be extended to class-specific alpha values
            weighted_loss = alpha * weighted_loss

        return weighted_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """Module wrapper for weighted cross entropy loss"""

    def __init__(self, weights=None, dataset=None, smooth_factor=0.1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.dataset = dataset
        self.smooth_factor = smooth_factor

        # Calculate weights immediately if dataset is provided
        if weights is None and dataset is not None:
            self.weights = WeightedLoss.get_class_weights(dataset)

    def forward(self, outputs, targets):
        return WeightedLoss.weighted_cross_entropy_loss(
            outputs, targets, self.weights, self.dataset, self.smooth_factor)


class AggressiveMinorityWeightedLoss(nn.Module):
    """Module wrapper for aggressive minority weighted loss"""

    def __init__(self, dataset=None, smooth_factor=0.1):
        super(AggressiveMinorityWeightedLoss, self).__init__()
        self.dataset = dataset
        self.smooth_factor = smooth_factor
        self.weights = None

        # Calculate weights immediately if dataset is provided
        if dataset is not None:
            self.weights = WeightedLoss.get_class_weights(
                dataset, method='squared_inverse', scaling='log', smooth_factor=smooth_factor)

    def forward(self, outputs, targets):
        if self.weights is not None:
            return F.cross_entropy(outputs, targets, weight=self.weights.to(targets.device))
        return WeightedLoss.aggressive_minority_weighted_loss(
            outputs, targets, self.dataset, self.smooth_factor)


class DynamicSampleWeightedLoss(nn.Module):
    """Module wrapper for dynamic sample weighted loss (focal loss with dynamic gamma)"""

    def __init__(self, max_epochs, alpha=0.5, gamma=2.0):
        super(DynamicSampleWeightedLoss, self).__init__()
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.gamma = gamma
        self.epoch = 0

    def forward(self, outputs, targets):
        return WeightedLoss.dynamic_sample_weighting(
            outputs, targets, self.epoch, self.max_epochs, self.alpha, self.gamma)

    def update_epoch(self, epoch):
        """Update current epoch number"""
        self.epoch = epoch
        gamma_val = self.gamma * (1 + min(epoch / self.max_epochs, 1.0))
        logging.debug(f"Dynamic loss: epoch {epoch}, gamma {gamma_val:.2f}")
