import time
import logging


class Timer:
    def __init__(self):
        self.start_time = None
        logging.info("Timer initialized.")

    @staticmethod
    def format_duration(seconds):
        """
        Format duration from seconds to hours, minutes, and seconds.

        Args:
        - seconds (int): Duration in seconds.

        Returns:
        - Formatted duration string.
        """
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        h, m, s = int(h), int(m), int(s)
        hour_str = f"{h} {'hr' if h == 1 else 'hrs'}"
        min_str = f"{m} {'min' if m == 1 else 'mins'}"
        sec_str = f"{s} {'sec' if s == 1 else 'secs'}"
        # duration_str = f"{hour_str}, {min_str}, {sec_str}"
        duration_str = f"{min_str}, {sec_str}"

        return duration_str

    @staticmethod
    def early_stopping(patience, validation_losses):
        if len(validation_losses) > patience and validation_losses[-1] > min(validation_losses[-patience - 1:-1]):
            logging.info("Early stopping triggered.")
            return True
        return False
