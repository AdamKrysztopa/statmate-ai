"""Logging configuration for StatMate AI.

This module sets up structured logging with appropriate handlers and formatters.
"""

import logging
import sys

from statmate.config import LoggingConfig


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Set up logging configuration for StatMate.

    Args:
        config: Logging configuration. If None, uses default configuration.

    Returns:
        Configured logger instance.
    """
    if config is None:
        config = LoggingConfig()

    # Get or create logger
    logger = logging.getLogger('statmate')
    logger.setLevel(getattr(logging, config.level))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if config.log_to_file:
        file_handler = logging.FileHandler(config.log_file_path)
        file_handler.setLevel(getattr(logging, config.level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Suppress HTTPX verbose logging if configured
    if config.suppress_httpx:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)

    logger.info('StatMate logging initialized')
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Name of the module (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f'statmate.{name}')


# Initialize default logger
default_logger = setup_logging()
