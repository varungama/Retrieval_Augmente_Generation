import os
import logging
from datetime import datetime
import inspect


# Function to create directory structure based on year/month/day
def create_log_directory(base_dir="logs"):
    """Create directories for logs separated by year, month, and day."""
    current_date = datetime.now()
    log_path = os.path.join(base_dir,
                            str(current_date.year),
                            f"{current_date.month:02d}",
                            f"{current_date.day:02d}")

    # Create the directories if they don't exist
    os.makedirs(log_path, exist_ok=True)

    # Create the log filename based on the current date and module name
    log_filename = os.path.join(log_path, f"{datetime.now().strftime('%H_%M_%S')}.log")

    return log_filename


# Function to configure logging
def get_logger(log_filename):
    """Setup logger using the calling module's filename."""

    # Get the name of the calling module (script file)
    frame = inspect.stack()[1]
    module_name = os.path.splitext(os.path.basename(frame.filename))[0]  # Get script file name without extension

    # Configure the logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging to file (in append mode)
    file_handler = logging.FileHandler(log_filename, mode='a')  # 'a' ensures append mode
    file_handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:  # Avoid adding handlers multiple times in case of reimport
        logger.addHandler(file_handler)

    return logger
