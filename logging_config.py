import logging

def setup_logger(log_file="app.log", level=logging.INFO):
    """
    Set up and return a logger to be shared across modules.

    Args:
        log_file (str): Path to the log file.
        level (int): Logging level, e.g., logging.DEBUG, logging.INFO.
    """
    # Create logger
    logger = logging.getLogger("shared_logger")
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
