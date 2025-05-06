import logging
import os

def setup_logging(logger_name, log_file=None, level=logging.INFO):
    """
    Creates and returns a logger with the specified name.
    If `log_file` is None, only logs to console (used in testing).
    """
    logger = logging.getLogger(logger_name)

    # Remove any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = False

    # Disable logging if running tests
    if os.getenv('PYTEST_CURRENT_TEST'):
        logger.setLevel(logging.CRITICAL)  # suppress all logs
        return logger

    # Formatters
    console_formatter = logging.Formatter('%(filename)s - line %(lineno)d - %(levelname)s - %(message)s')
    logfile_formatter = logging.Formatter('%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s')

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Optionally add file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logfile_formatter)
        logger.addHandler(file_handler)

    return logger
