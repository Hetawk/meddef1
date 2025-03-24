import os
import logging


def setup_logger(log_file=None, task_name=None, dataset_name=None, model_name=None):
    """
    Set up the logger with the specified log file path or construct one from components.

    Args:
        log_file: Optional direct path to log file
        task_name: Task name for constructing path
        dataset_name: Dataset name for constructing path
        model_name: Model name for constructing path
    """
    # If specific components are provided, construct the path
    if task_name and dataset_name and model_name:
        log_dir = os.path.join('out', task_name, dataset_name, model_name)
        log_file = os.path.join(log_dir, 'training.log')
    elif task_name and dataset_name:
        log_dir = os.path.join('out', task_name, dataset_name)
        log_file = os.path.join(log_dir, 'training.log')
    elif task_name:
        log_dir = os.path.join('out', task_name)
        log_file = os.path.join(log_dir, 'training.log')
    else:
        # Default fallback
        log_file = log_file or 'out/logger.txt'
        log_dir = os.path.dirname(log_file)

    # Ensure directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent the root logger from logging messages

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    # Remove timestamp (%(asctime)s) from the format
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info(f"Logger initialized. Logging to: {log_file}")

    return logger
