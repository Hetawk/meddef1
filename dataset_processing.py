import argparse
import logging
from pathlib import Path
from loader.dataset_handler import DatasetHandler
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Processing Tool')
    parser.add_argument('--datasets', nargs='+',
                        help='List of dataset names to process')
    parser.add_argument('--output_dir', default='processed_data',
                        help='Output directory for processed datasets')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.datasets:
        logging.error("No datasets specified")
        return

    for dataset_name in args.datasets:
        try:
            logging.info(f"Processing dataset: {dataset_name}")
            handler = DatasetHandler(
                dataset_name,
                config_path=os.path.join(os.path.dirname(
                    __file__), 'loader', 'config.yaml')
            )
            handler.process_and_load(
                args.output_dir,
                train_batch_size=32,
                val_batch_size=32,
                test_batch_size=32
            )
            logging.info(f"Completed processing {dataset_name}")
        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {str(e)}")
            logging.debug("Exception details:", exc_info=True)


if __name__ == "__main__":
    main()
