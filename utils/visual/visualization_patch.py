import logging


def patch_visualization():
    """Apply patches to the Visualization class to prevent warnings"""
    try:
        from utils.visual.visualization import Visualization

        # Store the original visualize_metrics method
        original_visualize_metrics = Visualization.visualize_metrics

        # Create a wrapper that handles empty data
        def safe_visualize_metrics(self, metrics, task_name, dataset_name, model_name,
                                   phase="train", class_names=None):
            try:
                # Check if metrics contain data
                if not metrics or 'per_class' not in metrics or not metrics['per_class']:
                    logging.info(
                        f"No metrics data available for {model_name}. Skipping visualization.")
                    return None

                # Call original method
                return original_visualize_metrics(self, metrics, task_name, dataset_name,
                                                  model_name, phase, class_names)

            except Exception as e:
                logging.warning(f"Error in visualize_metrics (patched): {e}")
                return None

        # Replace the method with our patched version
        Visualization.visualize_metrics = safe_visualize_metrics
        logging.debug("Successfully patched Visualization.visualize_metrics")

        return True
    except Exception as e:
        logging.warning(f"Failed to patch Visualization class: {e}")
        return False
