import logging

def setup_logging(logger_name, log_file='logs/scraper.log', level=logging.INFO):
    """
    Creates and returns a logger with the specified name.
    Ensures no duplicate handlers are added and the root logger is not triggered.
    """
    logger = logging.getLogger(logger_name)

    # Remove any pre-existing handlers attached to the logger (to prevent duplicate outputs)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Disable propagation to avoid root logger duplications
    logger.propagate = False

    # Create formatter with line number and filename
    logfile_formatter = logging.Formatter('%(asctime)s - %(filename)s - line %(lineno)d - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(filename)s - line %(lineno)d - %(levelname)s - %(message)s')

    # Create file handler (logs to file)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(logfile_formatter)

    # Create stream handler (logs to console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
