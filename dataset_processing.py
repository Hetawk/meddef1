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
    parser.add_argument('--enforce_split', action='store_true',
                        help='Enforce consistent dataset split ratios (default: True)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training data split ratio (default: 0.8)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation data split ratio (default: 0.1)')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Test data split ratio (default: 0.1)')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if not args.datasets:
        logging.error("No datasets specified")
        return

    # Validate split ratios sum to 1.0
    total_ratio = args.train_split + args.val_split + args.test_split
    if abs(total_ratio - 1.0) > 0.001:
        logging.warning(
            f"Split ratios sum to {total_ratio}, not 1.0. Normalizing...")
        factor = 1.0 / total_ratio
        args.train_split *= factor
        args.val_split *= factor
        args.test_split *= factor
        logging.info(
            f"Adjusted splits: Train={args.train_split:.2f}, Val={args.val_split:.2f}, Test={args.test_split:.2f}")

    split_ratio = (args.train_split, args.val_split, args.test_split)

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
                test_batch_size=32,
                enforce_split_ratio=args.enforce_split,
                split_ratio=split_ratio
            )
            logging.info(f"Completed processing {dataset_name}")
        except Exception as e:
            logging.error(f"Error processing {dataset_name}: {str(e)}")
            logging.debug("Exception details:", exc_info=True)


if __name__ == "__main__":
    main()
