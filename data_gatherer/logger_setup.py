import logging
import os
import sys

def setup_logging(logger_name, log_file=None, level=logging.WARNING, clear_previous_logs=False):
    """
    Creates and returns a logger with the specified name.
    If `log_file` is None, only logs to console (used in testing).
    """
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger(logger_name)

    # Remove any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    # Always show logs, even during tests (do not suppress logging)

    WHITE = "\033[97m"
    RESET = "\033[0m"
    console_format = f"{WHITE}%(filename)s - line %(lineno)d - %(levelname)s - %(message)s{RESET}"

    # Formatters
    console_formatter = logging.Formatter(console_format)
    logfile_formatter = logging.Formatter('%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s')

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optionally add file handler
    if log_file:
        if clear_previous_logs and os.path.exists(log_file):
            for f in os.listdir(os.path.dirname(log_file)):
                if f.endswith('.log'):
                    os.remove(os.path.join(os.path.dirname(log_file), f))

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logfile_formatter)
        logger.addHandler(file_handler)

    return logger
