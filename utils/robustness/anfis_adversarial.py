import numpy as np
import torch
import logging
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Tuple, List, Optional


class ANFISAdversarialTrainer:
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS) for enhancing adversarial training.

    This system uses fuzzy logic to determine sample weights for adversarial examples
    based on their sensitivity to perturbations and perturbation magnitude.

    Particularly useful for medical images where subtle features are important.
    """

    def __init__(self, model, attack_eps=0.1, attack_steps=7, attack_alpha=0.01,
                 sensitivity_weight=0.7, perturbation_weight=0.3):
        """
        Initialize the ANFIS adversarial trainer.

        Args:
            model: The model being trained
            attack_eps: Maximum perturbation size (epsilon)
            attack_steps: Number of steps for iterative attacks
            attack_alpha: Step size for iterative attacks
            sensitivity_weight: Weight for sensitivity component (prediction differences)
            perturbation_weight: Weight for perturbation magnitude component
        """
        self.model = model
        self.attack_eps = attack_eps
        self.attack_steps = attack_steps
        self.attack_alpha = attack_alpha
        self.sensitivity_weight = sensitivity_weight
        self.perturbation_weight = perturbation_weight

        try:
            self.fuzzy_system = self._create_fuzzy_system()
            self.fuzzy_enabled = True
            logging.info("ANFIS fuzzy system initialized successfully")
        except ImportError:
            logging.warning(
                "scikit-fuzzy not installed. Falling back to direct calculation.")
            self.fuzzy_enabled = False
        except Exception as e:
            logging.error(f"Error initializing fuzzy system: {e}")
            self.fuzzy_enabled = False

    def _create_fuzzy_system(self) -> ctrl.ControlSystemSimulation:
        """Create a fuzzy inference system for weighting samples"""
        # Create fuzzy variables
        sensitivity = ctrl.Antecedent(np.linspace(0, 1, 100), 'sensitivity')
        perturbation = ctrl.Antecedent(np.linspace(0, 1, 100), 'perturbation')
        weight = ctrl.Consequent(np.linspace(0, 1, 100), 'sample_weight')

        # Define membership functions
        sensitivity['low'] = fuzz.trimf(sensitivity.universe, [0, 0, 0.5])
        sensitivity['medium'] = fuzz.trimf(sensitivity.universe, [0, 0.5, 1])
        sensitivity['high'] = fuzz.trimf(sensitivity.universe, [0.5, 1, 1])

        perturbation['small'] = fuzz.trimf(perturbation.universe, [0, 0, 0.3])
        perturbation['medium'] = fuzz.trimf(
            perturbation.universe, [0.2, 0.5, 0.8])
        perturbation['large'] = fuzz.trimf(perturbation.universe, [0.7, 1, 1])

        weight['low'] = fuzz.trimf(weight.universe, [0, 0, 0.4])
        weight['medium'] = fuzz.trimf(weight.universe, [0.3, 0.5, 0.7])
        weight['high'] = fuzz.trimf(weight.universe, [0.6, 1, 1])

        # Define rules particularly useful for medical imaging
        # Very sensitive samples with small perturbation get high weight
        rule1 = ctrl.Rule(sensitivity['high'] &
                          perturbation['small'], weight['high'])
        # Medical features with medium perturbation still important
        rule2 = ctrl.Rule(sensitivity['high'] &
                          perturbation['medium'], weight['high'])
        # Large perturbations might distort features too much
        rule3 = ctrl.Rule(sensitivity['high'] &
                          perturbation['large'], weight['medium'])
        rule4 = ctrl.Rule(sensitivity['medium'] &
                          perturbation['small'], weight['medium'])
        rule5 = ctrl.Rule(sensitivity['medium'] &
                          perturbation['large'], weight['medium'])
        rule6 = ctrl.Rule(sensitivity['low'] &
                          perturbation['small'], weight['low'])
        rule7 = ctrl.Rule(sensitivity['low'] &
                          perturbation['large'], weight['low'])

        # Create control system
        system = ctrl.ControlSystem(
            [rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        return ctrl.ControlSystemSimulation(system)

    def calculate_sample_weights(self, images: torch.Tensor, adv_images: torch.Tensor,
                                 outputs: torch.Tensor, adv_outputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate weights for each sample based on sensitivity and perturbation magnitude

        Args:
            images: Original images (B x C x H x W)
            adv_images: Adversarial images (B x C x H x W)
            outputs: Model outputs for original images (B x num_classes)
            adv_outputs: Model outputs for adversarial images (B x num_classes)

        Returns:
            Tensor of sample weights (B)
        """
        batch_size = images.shape[0]
        weights = torch.ones(batch_size, device=images.device)

        # If no fuzzy system, return uniform weights
        if not self.fuzzy_enabled:
            return weights

        # Additional safety check for NaN inputs
        if torch.isnan(images).any() or torch.isnan(adv_images).any() or \
           torch.isnan(outputs).any() or torch.isnan(adv_outputs).any():
            logging.warning(
                "NaN detected in ANFIS inputs - using uniform weights")
            return weights

        # Process each sample in batch
        for i in range(batch_size):
            try:
                # Calculate prediction sensitivity with bounds
                pred_diff = torch.abs(
                    outputs[i] - adv_outputs[i]).mean().item()

                # Clip to reasonable range and check for NaN
                if np.isnan(pred_diff) or pred_diff > 10.0:
                    pred_diff = 1.0  # Default to high sensitivity if NaN
                else:
                    pred_diff = min(pred_diff, 1.0)  # Cap at 1.0

                # Calculate perturbation magnitude with safety checks
                pert_tensor = adv_images[i] - images[i]
                if torch.isnan(pert_tensor).any():
                    # If perturbation has NaN, use default weight
                    weights[i] = 0.5
                    continue

                pert_magnitude = torch.norm(
                    pert_tensor).item() / images[i].numel()

                # Handle zero or negative magnitudes
                if pert_magnitude <= 0 or np.isnan(pert_magnitude):
                    pert_magnitude = 1e-6  # Small but non-zero

                # Normalize perturbation relative to epsilon
                norm_pert = min(pert_magnitude / (self.attack_eps + 1e-8), 1.0)

                # Medical images need special handling for small perturbations
                if norm_pert < 0.05:
                    # For very small perturbations, linear weight based on sensitivity
                    weights[i] = 0.3 + 0.7 * pred_diff
                    continue

                # Apply fuzzy rules with error handling
                try:
                    self.fuzzy_system.input['sensitivity'] = pred_diff
                    self.fuzzy_system.input['perturbation'] = norm_pert

                    # Compute fuzzy output
                    self.fuzzy_system.compute()
                    weight_val = self.fuzzy_system.output['sample_weight']

                    # Sanity check on output
                    if np.isnan(weight_val) or weight_val <= 0:
                        raise ValueError("Invalid weight produced")

                    # Bound weights between 0.1 and 3.0
                    weights[i] = min(max(weight_val, 0.1), 3.0)
                except Exception as e:
                    logging.warning(f"Error in ANFIS fuzzy computation: {e}")
                    # Fall back to direct calculation
                    weights[i] = self._direct_calculation(images[i], adv_images[i],
                                                          outputs[i], adv_outputs[i])
            except Exception as e:
                logging.warning(
                    f"Error in ANFIS calculation for sample {i}: {e}")
                # Fall back to moderate weight
                weights[i] = 0.7

        # Normalize weights to avoid extreme values
        if torch.max(weights) > 0:
            # Clip extreme values
            weights = torch.clamp(weights, 0.2, 2.0)
            # Normalize to mean=1 but keep relative differences
            weights = weights / (weights.mean() + 1e-8)

        return weights

    def _direct_calculation(self, image: torch.Tensor, adv_image: torch.Tensor,
                            output: torch.Tensor, adv_output: torch.Tensor) -> float:
        """Direct calculation as fallback if fuzzy system fails"""
        # Calculate prediction sensitivity
        pred_diff = torch.abs(output - adv_output).mean().item()

        # Calculate perturbation magnitude
        pert_magnitude = torch.norm(image - adv_image).item() / image.numel()
        norm_pert = min(pert_magnitude / self.attack_eps, 1.0)

        # Simple weighted combination
        return (self.sensitivity_weight * pred_diff +
                self.perturbation_weight * (1 - norm_pert))

    def weight_adversarial_loss(self, loss: torch.Tensor, images: torch.Tensor,
                                adv_images: torch.Tensor, outputs: torch.Tensor,
                                adv_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply ANFIS-based weighting to the adversarial loss

        Args:
            loss: Per-sample adversarial loss values (B)
            images, adv_images, outputs, adv_outputs: As in calculate_sample_weights()

        Returns:
            Weighted adversarial loss
        """
        # Safety check for NaN in loss
        if torch.isnan(loss).any():
            logging.warning(
                "NaN detected in loss input to ANFIS - using mean instead")
            valid_loss = loss[~torch.isnan(loss)]
            if len(valid_loss) > 0:
                return valid_loss.mean()
            else:
                return torch.tensor(1.0, device=loss.device)

        try:
            # More defensive input checking
            if hasattr(self, 'fuzzy_system') and self.fuzzy_enabled:
                # Calculate sample weights
                weights = self.calculate_sample_weights(
                    images, adv_images, outputs, adv_outputs)

                # Apply weights to loss with safety check
                weighted_loss = loss * weights

                # Check for NaN after weighting
                if torch.isnan(weighted_loss).any():
                    logging.warning(
                        "ANFIS produced NaN weighted loss - using unweighted loss")
                    return loss.mean()

                return weighted_loss.mean()
            else:
                # If fuzzy system isn't available, use simple mean
                return loss.mean()
        except Exception as e:
            logging.warning(f"Error in ANFIS loss weighting: {e}")
            # Fall back to unweighted loss
            return loss.mean()
