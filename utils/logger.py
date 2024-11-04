import os
import logging

def setup_logger(log_file):
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logger initialized.")