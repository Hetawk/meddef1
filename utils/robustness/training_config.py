import logging
import math
import torch


class AdversarialTrainingConfig:
    """Configuration helper for adversarial training"""

    @staticmethod
    def get_epsilon_schedule(initial_epsilon, final_epsilon, epsilon_steps, epoch, schedule_type='linear'):
        """
        Calculate epsilon for the current epoch based on schedule type

        Args:
            initial_epsilon: Starting epsilon value
            final_epsilon: Final epsilon value
            epsilon_steps: Number of steps to reach final epsilon
            epoch: Current epoch
            schedule_type: Type of schedule ('linear', 'exponential', 'cosine')

        Returns:
            Current epsilon value
        """
        if epoch >= epsilon_steps:
            return final_epsilon

        progress = epoch / epsilon_steps

        if schedule_type == 'linear':
            # Linear schedule (current implementation)
            return initial_epsilon + (final_epsilon - initial_epsilon) * progress

        elif schedule_type == 'exponential':
            # Exponential schedule - slower at start, faster later
            return initial_epsilon * math.pow(final_epsilon / initial_epsilon, progress)

        elif schedule_type == 'cosine':
            # Cosine schedule - smooth transition
            return initial_epsilon + (final_epsilon - initial_epsilon) * 0.5 * (1 - math.cos(math.pi * progress))

        else:
            logging.warning(
                f"Unknown schedule type: {schedule_type}, using linear")
            return initial_epsilon + (final_epsilon - initial_epsilon) * progress

    @staticmethod
    def get_pgd_alpha(epsilon, steps, multiplier=2.5):
        """
        Calculate appropriate alpha (step size) for PGD attack based on epsilon

        Args:
            epsilon: Current epsilon value
            steps: Number of PGD steps
            multiplier: Controls the aggressiveness of steps

        Returns:
            Appropriate alpha value
        """
        # Common practice is to set alpha = epsilon / steps * multiplier
        return min(epsilon * multiplier / steps, 0.01)

    @staticmethod
    def get_combined_loss_weights(current_epoch, warmup_epochs=5, final_adv_weight=0.5):
        """
        Get clean and adversarial loss weights that sum to 1.0

        Args:
            current_epoch: Current training epoch
            warmup_epochs: Epochs of clean-focused training
            final_adv_weight: Final weight for adversarial loss

        Returns:
            Tuple of (clean_weight, adv_weight)
        """
        # During warmup, emphasize clean accuracy
        if current_epoch < warmup_epochs:
            clean_weight = 0.8 - (0.3 * current_epoch / warmup_epochs)
            adv_weight = 1.0 - clean_weight
        else:
            # After warmup, gradually reach final adversarial weight
            progress = min((current_epoch - warmup_epochs) / 10.0, 1.0)
            adv_weight = min(0.5 + (final_adv_weight - 0.5)
                             * progress, final_adv_weight)
            clean_weight = 1.0 - adv_weight

        return clean_weight, adv_weight
