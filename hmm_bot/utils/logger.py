"""
utils/logger.py — Shared rotating logger factory.

Call setup_logger(name) from any module to get a named logger that
writes to both logs/bot.log and the console.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Default log file relative to hmm_bot/ root
_DEFAULT_LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "logs", "bot.log"
)


def setup_logger(
    name: str = "HMMBot",
    log_file: str = _DEFAULT_LOG_FILE,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Return a named logger with rotating file + console handlers.

    Args:
        name:     Logger name (shown in each log line).
        log_file: Absolute or relative path to the log file.
        level:    Logging level (default INFO).

    Returns:
        logging.Logger instance.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicating handlers when called multiple times
    if not logger.handlers:
        # Rotating file handler: 5 MB × 5 backups
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
