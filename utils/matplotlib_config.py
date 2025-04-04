import matplotlib
import logging
import os
import sys


def configure_matplotlib_backend():
    """Configure matplotlib backend appropriately for the environment"""
    try:
        # Set environment variable to avoid issues
        os.environ['MPLCONFIGDIR'] = os.path.join(
            os.path.expanduser('~'), '.matplotlib')

        # Force non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        logging.info("Setting matplotlib backend to 'Agg'")

        # Import pyplot after backend is set
        import matplotlib.pyplot as plt

        # Configure to suppress warnings
        plt.rcParams['figure.max_open_warning'] = 0

        # Set default DPI for better image quality
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150

        # Disable info logging from matplotlib
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.category').setLevel(logging.ERROR)

        return True
    except Exception as e:
        logging.error(f"Error configuring matplotlib: {e}")
        return False
