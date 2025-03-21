"""
Wrapper for TensorBoard that handles compatibility issues across Python versions.
"""
import logging
import os
import warnings
from datetime import datetime

# Use a complete try/except block for TensorBoard import
TENSORBOARD_AVAILABLE = False
try:
    # First try importing tensorboard directly
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
    logging.info(
        f"Successfully imported TensorBoard v{tensorboard.__version__}")
except (ImportError, AttributeError) as e:
    # If that fails, try a manual approach
    try:
        # Try to use packaging directly
        import packaging.version
        import sys

        # Set up a compatibility layer
        if 'distutils' not in sys.modules:
            sys.modules['distutils'] = type('distutils', (), {})
        if not hasattr(sys.modules['distutils'], 'version'):
            sys.modules['distutils'].version = type('version', (), {})
        sys.modules['distutils'].version.LooseVersion = packaging.version.Version

        # Now try importing TensorBoard again
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
        logging.info(
            "Successfully imported TensorBoard using packaging compatibility")
    except Exception as e2:
        warnings.warn(
            f"TensorBoard not available: {str(e2)}. Please install 'packaging' package with: pip install packaging")

        # Define a dummy SummaryWriter for graceful degradation
        class SummaryWriter:
            def __init__(self, log_dir=None):
                self.log_dir = log_dir
                logging.warning(
                    f"TensorBoard not available. Would have logged to {log_dir}")

            def add_scalar(self, *args, **kwargs):
                pass

            def add_scalars(self, *args, **kwargs):
                pass

            def add_graph(self, *args, **kwargs):
                pass

            def add_hparams(self, *args, **kwargs):
                pass

            def add_histogram(self, *args, **kwargs):
                pass

            def close(self):
                pass


class TensorBoardLogger:
    """
    Wrapper class for TensorBoard SummaryWriter with graceful fallback
    and additional convenience methods.
    """

    def __init__(self, task_name, dataset_name, model_name):
        """Initialize TensorBoard logger with appropriate log directory."""
        if TENSORBOARD_AVAILABLE:
            log_dir = os.path.join('out/runs', task_name, dataset_name, model_name,
                                   datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.writer = SummaryWriter(log_dir)
            logging.info(f"TensorBoard logs will be saved to {log_dir}")
        else:
            self.writer = SummaryWriter()  # Dummy writer

    def log_hparams(self, hparams):
        """Log hyperparameters."""
        self.writer.add_hparams(hparams, {'hparam/initialized': 1})

    def log_model_graph(self, model, sample_input):
        """Log model graph with sample input."""
        try:
            self.writer.add_graph(model, sample_input)
        except Exception as e:
            logging.warning(
                f"Could not add model graph to TensorBoard: {str(e)}")

    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc,
                          learning_rate, adv_loss=None, adv_acc=None):
        """Log metrics for an epoch."""
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Accuracy/train', train_acc, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('Learning_rate', learning_rate, epoch)

        if adv_loss is not None:
            self.writer.add_scalar('Loss/train_adv', adv_loss, epoch)
        if adv_acc is not None:
            self.writer.add_scalar('Accuracy/train_adv', adv_acc, epoch)

    def log_test_results(self, test_loss, test_acc, adv_test_loss=None, adv_test_acc=None):
        """Log final test results."""
        self.writer.add_scalar('Loss/test', test_loss, 0)
        self.writer.add_scalar('Accuracy/test', test_acc, 0)
        if adv_test_loss is not None:
            self.writer.add_scalar('Loss/test_adv', adv_test_loss, 0)
        if adv_test_acc is not None:
            self.writer.add_scalar('Accuracy/test_adv', adv_test_acc, 0)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
