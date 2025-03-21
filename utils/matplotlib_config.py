import matplotlib
import logging
import os
import sys


def configure_matplotlib_backend():
    """
    Configure matplotlib backend to avoid tkinter thread errors.

    This should be called at the start of the application before
    any other matplotlib imports or usage.
    """
    # Check if we're running in interactive mode or in a script
    is_interactive = hasattr(sys, 'ps1')

    # Determine appropriate backend
    if os.environ.get('DISPLAY') is None or 'pytest' in sys.modules:
        # No display available or running in test environment
        backend = 'Agg'  # Non-interactive backend
    elif not is_interactive:
        # Running as a script - use Agg to avoid thread issues
        backend = 'Agg'
    else:
        # Interactive mode - let matplotlib choose
        backend = None

    # Set backend if specified
    if backend:
        logging.info(f"Setting matplotlib backend to '{backend}'")
        matplotlib.use(backend, force=True)

    # Return selected backend
    return matplotlib.get_backend()
