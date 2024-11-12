import os
import logging

def setup_logger(log_file):
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.info("Logger initialized.")