"""
A simple metrics logger that provides similar functionality to TensorBoard
but works without any dependencies.
"""
import os
import json
import logging
import csv
from datetime import datetime
import matplotlib.pyplot as plt

# Optional TensorBoard Support
TENSORBOARD_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        logging.info(
            "TensorBoard/TensorboardX not available, using standalone metrics logging")


class MetricsLogger:
    """
    Simple alternative to TensorBoard that logs metrics to files
    and creates plots.
    """

    def __init__(self, task_name, dataset_name, model_name, use_tensorboard=True):
        """Initialize logger with appropriate log directory."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(
            'runs', task_name, dataset_name, model_name, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.json_file = os.path.join(self.log_dir, 'metrics.json')
        self.csv_file = os.path.join(self.log_dir, 'metrics.csv')

        # TensorBoard support
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.use_tensorboard:
            try:
                self.writer = SummaryWriter(log_dir=self.log_dir)
                logging.info(f"TensorBoard logging enabled at {self.log_dir}")
            except Exception as e:
                logging.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False
                self.writer = None

        # Initialize metrics storage
        self.metrics = {
            'hparams': {},
            'epochs': [],
            'train': {'loss': [], 'accuracy': []},
            'val': {'loss': [], 'accuracy': []},
            'test': {'loss': None, 'accuracy': None},
            'learning_rate': [],
            'adversarial': {
                'train': {'loss': [], 'accuracy': []},
                'test': {'loss': None, 'accuracy': None}
            }
        }

        # Initialize CSV file
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                            'lr', 'adv_train_loss', 'adv_train_acc'])

        logging.info(f"Metrics will be saved to {self.log_dir}")

    def log_hparams(self, hparams):
        """Log hyperparameters."""
        self.metrics['hparams'] = hparams
        with open(os.path.join(self.log_dir, 'hparams.json'), 'w') as f:
            json.dump(hparams, f, indent=2)

        # Log to TensorBoard if available
        if self.use_tensorboard and self.writer is not None:
            # Convert hparams to compatible format (TensorBoard expects simple types)
            tb_hparams = {}
            for k, v in hparams.items():
                if isinstance(v, (int, float, str, bool)):
                    tb_hparams[k] = v
                else:
                    tb_hparams[k] = str(v)
            try:
                self.writer.add_hparams(tb_hparams, {})
            except Exception as e:
                logging.warning(f"Failed to log hparams to TensorBoard: {e}")

    def log_model_graph(self, model, sample_input):
        """Placeholder for model graph logging."""
        # We don't actually save the model graph, just a summary
        try:
            param_size = sum(p.numel() for p in model.parameters()) / 1e6
            with open(os.path.join(self.log_dir, 'model_summary.txt'), 'w') as f:
                f.write(f"Model: {type(model).__name__}\n")
                f.write(f"Parameters: {param_size:.2f}M\n")

            # Add to TensorBoard if available
            if self.use_tensorboard and self.writer is not None:
                try:
                    self.writer.add_graph(model, sample_input)
                except Exception as e:
                    logging.warning(
                        f"Failed to add model graph to TensorBoard: {e}")
        except Exception as e:
            logging.warning(f"Could not log model summary: {str(e)}")

    def log_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc,
                          learning_rate, adv_loss=None, adv_acc=None):
        """Log metrics for an epoch."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train']['loss'].append(train_loss)
        self.metrics['train']['accuracy'].append(train_acc)
        self.metrics['val']['loss'].append(val_loss)
        self.metrics['val']['accuracy'].append(val_acc)
        self.metrics['learning_rate'].append(learning_rate)

        if adv_loss is not None:
            self.metrics['adversarial']['train']['loss'].append(adv_loss)
        if adv_acc is not None:
            self.metrics['adversarial']['train']['accuracy'].append(adv_acc)

        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, learning_rate,
                             adv_loss if adv_loss is not None else '',
                             adv_acc if adv_acc is not None else ''])

        # Log to TensorBoard if available
        if self.use_tensorboard and self.writer is not None:
            try:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_rate', learning_rate, epoch)

                if adv_loss is not None:
                    self.writer.add_scalar('Loss/adv_train', adv_loss, epoch)
                if adv_acc is not None:
                    self.writer.add_scalar(
                        'Accuracy/adv_train', adv_acc, epoch)
            except Exception as e:
                logging.warning(
                    f"Failed to log epoch metrics to TensorBoard: {e}")

        # Update JSON file
        with open(self.json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Generate plots
        self._generate_plots()

    def log_test_results(self, test_loss, test_acc, adv_test_loss=None, adv_test_acc=None):
        """Log final test results."""
        self.metrics['test']['loss'] = test_loss
        self.metrics['test']['accuracy'] = test_acc

        if adv_test_loss is not None:
            self.metrics['adversarial']['test']['loss'] = adv_test_loss
        if adv_test_acc is not None:
            self.metrics['adversarial']['test']['accuracy'] = adv_test_acc

        # Write test results to a special file
        with open(os.path.join(self.log_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Test Accuracy: {test_acc}\n")
            if adv_test_loss is not None:
                f.write(f"Adversarial Test Loss: {adv_test_loss}\n")
            if adv_test_acc is not None:
                f.write(f"Adversarial Test Accuracy: {adv_test_acc}\n")

        # Log to TensorBoard if available
        if self.use_tensorboard and self.writer is not None:
            try:
                self.writer.add_scalar('Loss/test', test_loss, 0)
                self.writer.add_scalar('Accuracy/test', test_acc, 0)
                if adv_test_loss is not None:
                    self.writer.add_scalar('Loss/adv_test', adv_test_loss, 0)
                if adv_test_acc is not None:
                    self.writer.add_scalar(
                        'Accuracy/adv_test', adv_test_acc, 0)
            except Exception as e:
                logging.warning(
                    f"Failed to log test results to TensorBoard: {e}")

        # Update JSON file
        with open(self.json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def _generate_plots(self):
        """Generate and save plots for the current metrics."""
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epochs'], self.metrics['train']
                 ['loss'], 'b-', label='Training Loss')
        plt.plot(self.metrics['epochs'], self.metrics['val']
                 ['loss'], 'r-', label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close()

        # Plot training and validation accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epochs'], self.metrics['train']
                 ['accuracy'], 'b-', label='Training Accuracy')
        plt.plot(self.metrics['epochs'], self.metrics['val']
                 ['accuracy'], 'r-', label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'accuracy_plot.png'))
        plt.close()

        # Plot learning rate
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epochs'], self.metrics['learning_rate'], 'g-')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'lr_plot.png'))
        plt.close()

        # Plot adversarial metrics if available
        if self.metrics['adversarial']['train']['loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['epochs'], self.metrics['train']
                     ['accuracy'], 'b-', label='Clean Accuracy')
            plt.plot(self.metrics['epochs'], self.metrics['adversarial']
                     ['train']['accuracy'], 'r-', label='Adversarial Accuracy')
            plt.title('Clean vs Adversarial Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, 'adv_accuracy_plot.png'))
            plt.close()

    def close(self):
        """Clean up any resources."""
        # Save final metrics
        with open(self.json_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # Close TensorBoard writer if it exists
        if self.use_tensorboard and self.writer is not None:
            try:
                self.writer.close()
            except Exception as e:
                logging.warning(f"Error closing TensorBoard writer: {e}")

        # Generate final plots
        self._generate_plots()
