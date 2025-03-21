"""
Centralized logging configuration for GraphRAG using Loguru.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Remove default logger
logger.remove()

# Default log level (can be overridden with LOG_LEVEL env var)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Enable colors on Windows
os.environ["FORCE_COLOR"] = "1"

# Define log format
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Setup console handler
logger.configure(handlers=[
    {"sink": sys.stderr, "format": LOG_FORMAT, "level": LOG_LEVEL, "colorize": True}
])

# Setup file logging if LOG_FILE environment variable is defined
LOG_FILE = os.getenv("LOG_FILE")
if LOG_FILE:
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add rotating file handler
    logger.add(
        LOG_FILE,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        rotation="10 MB",  # Rotate when file reaches 10MB
        retention="1 week",  # Keep logs for 1 week
        compression="zip"  # Compress rotated logs
    )
    logger.info(f"Logging to file: {LOG_FILE}")

# Create a function to intercept standard library logging
def intercept_standard_logging():
    """
    Intercept and redirect standard library logging to Loguru.
    This is useful for third-party libraries using the standard logging module.
    """
    import logging
    
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where the logged message originated
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Configure standard logging to use our handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Update existing loggers to use our handler
    for name in logging.root.manager.loggerDict.keys():
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

# Intercept standard library logging
intercept_standard_logging()

# Export logger for use in other modules
__all__ = ["logger"] 